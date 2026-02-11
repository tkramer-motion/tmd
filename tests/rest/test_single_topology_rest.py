from collections.abc import Sequence
from dataclasses import replace
from functools import cache

import jax
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
from tmd.fe import atom_mapping
from tmd.fe.rbfe import setup_optimized_host
from tmd.fe.rest.bond import mkproper
from tmd.fe.rest.interpolation import (
    Exponential,
    InterpolationFxnName,
    Linear,
    Quadratic,
    Symmetric,
    plot_interpolation_fxn,
)
from tmd.fe.rest.single_topology import SingleTopologyREST
from tmd.fe.single_topology import SingleTopology
from tmd.fe.system import GuestSystem
from tmd.fe.utils import get_romol_conf, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.md import builders
from tmd.potentials import NonbondedInteractionGroup
from tmd.utils import path_to_internal_file

with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "ligands.sdf") as ligands_path:
    hif2a_ligands = read_sdf_mols_by_name(ligands_path)

hif2a_ligand_pairs = [
    (mol_a, mol_b)
    for mol_a_name, mol_a in hif2a_ligands.items()
    for mol_b_name, mol_b in hif2a_ligands.items()
    if mol_a_name < mol_b_name
]

forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")


@cache
def get_core(mol_a, mol_b) -> tuple[tuple[int, int], ...]:
    core_array = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    return tuple((a, b) for a, b in core_array)


@cache
def get_single_topology(mol_a, mol_b, core) -> SingleTopology:
    return SingleTopology(mol_a, mol_b, np.asarray(core), forcefield)


@cache
def get_single_topology_rest(
    mol_a, mol_b, core, max_temperature_scale: float, temperature_scale_interpolation_fxn: InterpolationFxnName
) -> SingleTopologyREST:
    return SingleTopologyREST(
        mol_a, mol_b, np.asarray(core), forcefield, max_temperature_scale, temperature_scale_interpolation_fxn
    )


@pytest.mark.parametrize("lamb", [0.0, 0.4, 0.5, 1.0])
@pytest.mark.parametrize("temperature_scale_interpolation_fxn", ["linear", "quadratic", "exponential"])
@pytest.mark.parametrize("mol_pair", np.random.default_rng(2024).choice(hif2a_ligand_pairs, size=3))
def test_single_topology_rest_vacuum(mol_pair, temperature_scale_interpolation_fxn, lamb):
    mol_a, mol_b = mol_pair

    has_aliphatic_rings = (
        rdMolDescriptors.CalcNumAliphaticRings(mol_a) > 0 or rdMolDescriptors.CalcNumAliphaticRings(mol_b) > 0
    )
    has_rotatable_bonds = (
        rdMolDescriptors.CalcNumRotatableBonds(mol_a) > 0 or rdMolDescriptors.CalcNumRotatableBonds(mol_b) > 0
    )

    box = np.eye(3) * 100.0
    core = get_core(mol_a, mol_b)
    st = get_single_topology(mol_a, mol_b, core)
    st_rest = get_single_topology_rest(mol_a, mol_b, core, 2.0, temperature_scale_interpolation_fxn)

    # NOTE: This assertion is not guaranteed to hold in general (i.e. the REST region may be empty, or the whole
    # combined ligand), but it does hold for typical cases, including the edges tested here. The stronger assertion here
    # ensures that later assertions (e.g. that we only soften interactions in the REST region) are not trivially true.
    assert 0 < len(st_rest.rest_region_atom_idxs) < st_rest.get_num_atoms()

    state = st_rest.setup_intermediate_state(lamb)
    state_ref = st.setup_intermediate_state(lamb)
    assert set(st_rest.propers) == set(mkproper(*idxs) for idxs in state_ref.proper.potential.idxs)
    assert set(st_rest.candidate_propers.values()) < set(st_rest.propers)

    ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))

    U_proper = state.proper(ligand_conf, box)
    U_proper_ref = state_ref.proper(ligand_conf, box)

    U_nonbonded = state.nonbonded_pair_list(ligand_conf, box)
    U_nonbonded_ref = state_ref.nonbonded_pair_list(ligand_conf, box)

    U = state.get_U_fn()(ligand_conf)
    U_ref = state_ref.get_U_fn()(ligand_conf)

    energy_scale = st_rest.get_energy_scale_factor(lamb)

    if lamb == 0.0 or lamb == 1.0:
        assert energy_scale == 1.0

        assert U_proper == U_proper_ref
        assert U_nonbonded == U_nonbonded_ref
        assert U == U_ref

    else:
        assert energy_scale < 1.0

        rest_atom_idxs = np.array(list(st_rest.rest_region_atom_idxs))
        rest_pair_pred = (
            (state.nonbonded_pair_list.potential.idxs[..., None] == rest_atom_idxs[None, None, :]).any(-1).any(-1)
        )

        def compute_lig_lig_ixn_energy(state: GuestSystem, pair_pred: NDArray[np.bool_]):
            nonbonded_pair_list = replace(
                state.nonbonded_pair_list,
                potential=replace(
                    state.nonbonded_pair_list.potential,
                    idxs=state.nonbonded_pair_list.potential.idxs[pair_pred, :],
                ),
                params=state.nonbonded_pair_list.params[pair_pred, :],
            )
            return nonbonded_pair_list(ligand_conf, box)

        # check that ligand-ligand pairs in the REST region are scaled appropriately
        U_nb_rest = compute_lig_lig_ixn_energy(state, rest_pair_pred)
        U_nb_rest_ref = compute_lig_lig_ixn_energy(state_ref, rest_pair_pred)
        np.testing.assert_allclose(U_nb_rest, energy_scale * U_nb_rest_ref)

        # check that ligand-ligand pairs in the REST region are not scaled
        complement_pair_pred = ~rest_pair_pred
        U_nb_complement = compute_lig_lig_ixn_energy(state, complement_pair_pred)
        U_nb_complement_ref = compute_lig_lig_ixn_energy(state_ref, complement_pair_pred)
        np.testing.assert_array_equal(U_nb_complement, U_nb_complement_ref)

        if has_rotatable_bonds or has_aliphatic_rings:
            assert 0 < len(st_rest.candidate_propers)
            assert not np.isclose(U_proper, U_proper_ref)
            assert U_proper < U_proper_ref

        def compute_proper_energy(state: GuestSystem, ixn_idxs: Sequence[int]):
            assert state.proper
            proper = replace(
                state.proper,
                potential=replace(state.proper.potential, idxs=state.proper.potential.idxs[ixn_idxs, :]),
                params=state.proper.params[ixn_idxs, :],
            )
            return proper(ligand_conf, box)

        # check that propers in the REST region are scaled appropriately
        rest_proper_idxs = st_rest.target_proper_idxs
        U_proper_rest = compute_proper_energy(state, rest_proper_idxs)
        U_proper_rest_ref = compute_proper_energy(state_ref, rest_proper_idxs)
        np.testing.assert_allclose(U_proper_rest, energy_scale * U_proper_rest_ref)

        # check that propers outside of the REST region are not scaled
        num_propers = len(st_rest.propers)
        complement_proper_idxs = set(range(num_propers)) - set(rest_proper_idxs)
        complement_proper_idxs = list(complement_proper_idxs)
        U_proper_complement = compute_proper_energy(state, complement_proper_idxs)
        U_proper_complement_ref = compute_proper_energy(state_ref, complement_proper_idxs)
        np.testing.assert_array_equal(U_proper_complement, U_proper_complement_ref)


@cache
def get_solvent_host(st: SingleTopology) -> tuple[builders.HostConfig, builders.HostConfig]:
    box_width = 4.0
    host_config = builders.build_water_system(box_width, st.ff.water_ff, mols=[st.mol_a, st.mol_b], box_margin=0.1)
    optimized_host_config = setup_optimized_host(host_config, [st.mol_a, st.mol_b], st.ff)

    return optimized_host_config, host_config


@pytest.mark.parametrize("lamb", [0.0, 0.4, 0.5, 1.0])
@pytest.mark.parametrize("temperature_scale_interpolation_fxn", ["linear", "quadratic", "exponential"])
@pytest.mark.parametrize(
    "mol_pair",
    [
        (hif2a_ligand_pairs[0][0], hif2a_ligand_pairs[0][0]),  # Identity transformation
        *np.random.default_rng(2024).choice(hif2a_ligand_pairs, size=3),  # Random pairs
    ],
)
def test_single_topology_rest_solvent(mol_pair, temperature_scale_interpolation_fxn, lamb):
    mol_a, mol_b = mol_pair

    core = get_core(mol_a, mol_b)
    st = get_single_topology(mol_a, mol_b, core)
    st_rest = get_single_topology_rest(mol_a, mol_b, core, 2.0, temperature_scale_interpolation_fxn)

    optimized_host, host_config = get_solvent_host(st)

    ligand_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    conf = np.concatenate([optimized_host.conf, ligand_conf])

    def compute_host_guest_ixn_energy(st: SingleTopology, ligand_idxs: set[int]):
        hgs = st.combine_with_host(
            optimized_host.host_system, lamb, host_config.num_water_atoms, host_config.omm_topology
        )
        num_atoms_host = optimized_host.host_system.nonbonded_all_pairs.potential.num_atoms
        ligand_idxs_ = np.array(list(ligand_idxs), dtype=np.int32) + num_atoms_host
        if len(ligand_idxs_) == 0:
            return 0  # If there are 0 ligand indices, the energy is zero

        return NonbondedInteractionGroup(
            hgs.nonbonded_all_pairs.potential.num_atoms,
            ligand_idxs_,
            hgs.nonbonded_all_pairs.potential.beta,
            hgs.nonbonded_all_pairs.potential.cutoff,
            col_atom_idxs=np.arange(num_atoms_host, dtype=np.int32),
        )(conf, hgs.nonbonded_all_pairs.params, host_config.box)

    # check that interactions involving atoms in the REST region are scaled appropriately
    U = compute_host_guest_ixn_energy(st_rest, st_rest.rest_region_atom_idxs)
    U_ref = compute_host_guest_ixn_energy(st, st_rest.rest_region_atom_idxs)
    energy_scale = st_rest.get_energy_scale_factor(lamb)
    np.testing.assert_allclose(U, energy_scale * U_ref, rtol=1e-5)

    # check that interactions involving atoms outside of the REST region are not scaled
    complement_atom_idxs = set(range(st_rest.get_num_atoms())) - st_rest.rest_region_atom_idxs
    U_complement = compute_host_guest_ixn_energy(st_rest, complement_atom_idxs)
    U_complement_ref = compute_host_guest_ixn_energy(st, complement_atom_idxs)
    np.testing.assert_array_equal(U_complement, U_complement_ref)


def get_mol(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=2024)
    return mol


def get_identity_transformation(mol):
    n_atoms = mol.GetNumAtoms()
    core = np.tile(np.arange(n_atoms)[:, None], (1, 2))  # identity
    return SingleTopologyREST(mol, mol, core, forcefield, 2.0, "linear")


def test_single_topology_rest_propers():
    """Example with some propers not in the REST region"""
    mol_a = hif2a_ligands["15"]
    mol_b = hif2a_ligands["30"]
    core = get_core(mol_a, mol_b)
    st = SingleTopologyREST(mol_a, mol_b, np.asarray(core), forcefield, 2.0)
    assert set(st.target_propers.items()) < set(st.candidate_propers.items())


def test_single_topology_rest_propers_identity():
    # benzene: no propers are scaled
    benzene = get_mol("c1ccccc1")
    st = get_identity_transformation(benzene)
    assert len(st.candidate_propers) == 0

    # cyclohexane: all 9 * 6 ring propers are scaled (|{H1, H2, C1}-C2-C3-{C4, H3, H4}| = 9 propers per C-C bond)
    cyclohexane = get_mol("C1CCCCC1")
    st = get_identity_transformation(cyclohexane)
    assert len(set(st.candidate_propers.values())) == 9 * 6

    # phenylcyclohexane: all 9 * 6 cyclohexane ring propers and 6 rotatable bond propers are scaled
    phenylcyclohexane = get_mol("c1ccc(C2CCCCC2)cc1")
    st = get_identity_transformation(phenylcyclohexane)
    assert len(set(st.candidate_propers.values())) == 9 * 6 + 6


@pytest.mark.parametrize(
    "lamb", [0.0, 0.4, 0.51, 1.0]
)  # NOTE: asymmetry at lambda = 0.5 due to discontinuity in combine_confs
@pytest.mark.parametrize("temperature_scale_interpolation_fxn", ["linear", "quadratic", "exponential"])
@pytest.mark.parametrize("mol_pair", np.random.default_rng(2024).choice(hif2a_ligand_pairs, size=3))
def test_single_topology_rest_symmetric(mol_pair, temperature_scale_interpolation_fxn, lamb):
    mol_a, mol_b = mol_pair
    core_fwd = get_core(mol_a, mol_b)
    core_rev = tuple((b, a) for a, b in core_fwd)

    def get_transformation(mol_a, mol_b, core, lamb):
        st = get_single_topology_rest(mol_a, mol_b, core, 2.0, temperature_scale_interpolation_fxn)
        potential = st.setup_intermediate_state(lamb).get_U_fn()
        conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb)
        return potential, conf, st

    u_fwd, conf_fwd, st_fwd = get_transformation(mol_a, mol_b, core_fwd, lamb)
    u_rev, conf_rev, st_rev = get_transformation(mol_b, mol_a, core_rev, 1.0 - lamb)

    assert len(st_fwd.rotatable_bonds) == len(st_rev.rotatable_bonds)
    assert len(st_fwd.aliphatic_ring_bonds) == len(st_rev.aliphatic_ring_bonds)
    assert len(st_fwd.target_proper_idxs) == len(st_rev.target_proper_idxs)

    np.testing.assert_allclose(u_fwd(conf_fwd), u_rev(conf_rev))

    core_to_a, core_to_b = np.asarray(core_fwd).T
    core_map = [(x, y) for x, y in zip(st_fwd.a_to_c, st_rev.b_to_c)]
    dummy_map = [(x, y) for x, y in zip(st_fwd.b_to_c, st_rev.a_to_c) if x not in core_to_a and y not in core_to_b]
    fused_map = core_map + dummy_map
    p_fwd, p_rev = np.array(fused_map).T

    conf_fwd = conf_fwd.astype(np.float32)
    conf_rev = conf_rev.astype(np.float32)

    np.testing.assert_allclose(
        jax.grad(u_fwd)(conf_fwd)[p_fwd],
        jax.grad(u_rev)(conf_rev)[p_rev],
        rtol=9e-5,  # 32bit
    )


def plot_interpolation_fxns():
    src, dst = 1.0, 3.0
    _ = plot_interpolation_fxn(Symmetric(Linear(src, dst)))
    _ = plot_interpolation_fxn(Symmetric(Quadratic(src, dst)))
    _ = plot_interpolation_fxn(Symmetric(Exponential(src, dst)))
    _ = plt.legend()
    plt.show()


# --- Tests for expand_rest_region_to_amides ---

from tmd.fe.rest.single_topology import get_amide_atoms_and_bonds
from tmd.graph_utils import convert_to_nx


def _get_atom_idx_by_map_num(mol, map_num):
    """Helper: find atom index by atom map number."""
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    raise ValueError(f"No atom with map number {map_num}")


def _get_atom_symbols(mol, idxs):
    """Helper: get set of (idx, symbol) for readable assertions."""
    return {(idx, mol.GetAtomWithIdx(idx).GetSymbol()) for idx in idxs}


def _find_amide_atoms(mol):
    """Helper: return (amide_n, carbonyl_c, carbonyl_o) for the first amide in mol."""
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3:1][CX3:2](=[OX1:3])")):
        return match
    raise ValueError("No amide found")


def test_expand_amides_includes_all_amide_atoms_from_c_side():
    """When BFS reaches the amide from the C(=O) side, ALL amide atoms (N, C, O) are included.

    Molecule: CH3-C(=O)-NH-CH3 (N-methylacetamide)
    Starting from the methyl C on the C(=O) side, we should get:
      methyl C + H's, C(=O), O, AND N -- but NOT the N-side methyl or its H's.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC"))
    nxg = convert_to_nx(mol)

    amide_n, carbonyl_c, carbonyl_o = _find_amide_atoms(mol)

    # Find the methyl C on the C(=O) side
    methyl_c_carbonyl_side = None
    for neighbor in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx != carbonyl_o and idx != amide_n and neighbor.GetSymbol() == "C":
            methyl_c_carbonyl_side = idx
    assert methyl_c_carbonyl_side is not None

    result = SingleTopologyREST.expand_rest_region_to_amides({methyl_c_carbonyl_side}, mol, nxg)

    # ALL amide atoms must be included
    assert carbonyl_c in result
    assert carbonyl_o in result
    assert amide_n in result

    # The methyl on the N side must NOT be in the result (past the amide boundary)
    methyl_c_n_side = None
    for neighbor in mol.GetAtomWithIdx(amide_n).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx != carbonyl_c and neighbor.GetSymbol() == "C":
            methyl_c_n_side = idx
    assert methyl_c_n_side is not None
    assert methyl_c_n_side not in result


def test_expand_amides_includes_all_amide_atoms_from_n_side():
    """When BFS reaches the amide from the N side, ALL amide atoms (N, C, O) are included.

    Molecule: CH3-C(=O)-NH-CH3
    Starting from the methyl on the N side, we should get:
      that methyl + H's, N, H on N, C(=O), AND O -- but NOT the C-side methyl or its H's.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC"))
    nxg = convert_to_nx(mol)

    amide_n, carbonyl_c, carbonyl_o = _find_amide_atoms(mol)

    # Find the methyl C on the N side
    methyl_c_n_side = None
    for neighbor in mol.GetAtomWithIdx(amide_n).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx != carbonyl_c and neighbor.GetSymbol() == "C":
            methyl_c_n_side = idx
    assert methyl_c_n_side is not None

    result = SingleTopologyREST.expand_rest_region_to_amides({methyl_c_n_side}, mol, nxg)

    # ALL amide atoms must be included
    assert amide_n in result
    assert carbonyl_c in result
    assert carbonyl_o in result

    # H on N should be included (BFS reaches N, traverses to H)
    n_hydrogens = [
        n.GetIdx() for n in mol.GetAtomWithIdx(amide_n).GetNeighbors() if n.GetSymbol() == "H"
    ]
    for h_idx in n_hydrogens:
        assert h_idx in result, f"H on N (idx={h_idx}) should be in expanded region"

    # The methyl on the C(=O) side must NOT be in the result (past the amide boundary)
    methyl_c_carbonyl_side = None
    for neighbor in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx != carbonyl_o and idx != amide_n and neighbor.GetSymbol() == "C":
            methyl_c_carbonyl_side = idx
    assert methyl_c_carbonyl_side is not None
    assert methyl_c_carbonyl_side not in result


def test_expand_amides_no_amide_returns_original():
    """With no amides, the function returns the original set unchanged (early return)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    nxg = convert_to_nx(mol)

    start = {0, 3}
    result = SingleTopologyREST.expand_rest_region_to_amides(start, mol, nxg)
    assert result == start


def test_expand_amides_stops_past_amide():
    """Expansion includes the full amide group but NOT atoms beyond it.

    Molecule: phenyl-C(=O)-NH-phenyl (benzanilide)
    Starting from one phenyl ring, the amide (N, C, O) is included but the
    other phenyl ring is not.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc(C(=O)Nc2ccccc2)cc1"))
    nxg = convert_to_nx(mol)

    amide_atoms, amide_bonds = get_amide_atoms_and_bonds(mol)
    assert len(amide_bonds) == 1

    amide_n, carbonyl_c, carbonyl_o = _find_amide_atoms(mol)

    # Find a ring atom on the C(=O) side
    c_side_ring_atom = None
    for neighbor in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx != carbonyl_o and idx != amide_n:
            c_side_ring_atom = idx
            break

    result = SingleTopologyREST.expand_rest_region_to_amides({c_side_ring_atom}, mol, nxg)

    # ALL amide atoms should be included
    assert carbonyl_c in result
    assert carbonyl_o in result
    assert amide_n in result

    # Atoms bonded to N on the far side (N-side ring) should NOT be included
    for neighbor in mol.GetAtomWithIdx(amide_n).GetNeighbors():
        if neighbor.GetIdx() != carbonyl_c:
            assert neighbor.GetIdx() not in result, (
                f"Atom {neighbor.GetIdx()} ({neighbor.GetSymbol()}) past the amide should not be included"
            )


def test_expand_amides_multiple_amides():
    """With two amides, BFS includes the nearest amide group but stops before the second.

    Molecule: CH3-C(=O)-NH-CH2-C(=O)-NH-CH3 (a dipeptide-like chain)
    Starting from the leftmost methyl, the first amide (N, C, O) is included
    but no atoms past the first amide N should be reached.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NCC(=O)NC"))
    nxg = convert_to_nx(mol)

    amide_atoms, amide_bonds = get_amide_atoms_and_bonds(mol)
    assert len(amide_bonds) == 2  # two amide N-C bonds

    # Get both amide groups
    query = Chem.MolFromSmarts("[NX3:1][CX3:2](=[OX1:3])")
    matches = mol.GetSubstructMatches(query)
    assert len(matches) == 2

    start = 0
    result = SingleTopologyREST.expand_rest_region_to_amides({start}, mol, nxg)

    # The first amide group (the one whose C is bonded to atom 0's neighbor) must be fully included
    first_amide_n, first_amide_c, first_amide_o = matches[0]
    assert first_amide_c in result
    assert first_amide_o in result
    assert first_amide_n in result

    # The second amide's atoms that are NOT shared with the first should NOT be in the result
    # (they're past the first amide boundary)
    second_amide_n, second_amide_c, second_amide_o = matches[1]
    # The CH2 between amides and atoms past it should not be reached
    # Find an atom only reachable by crossing the first amide
    assert second_amide_c not in result or second_amide_c == first_amide_c
    # The last methyl should definitely not be included
    last_methyl_c = None
    for neighbor in mol.GetAtomWithIdx(second_amide_n).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx != second_amide_c and neighbor.GetSymbol() == "C":
            last_methyl_c = idx
    if last_methyl_c is not None:
        assert last_methyl_c not in result


def test_expand_amides_starting_at_amide_atom():
    """If the initial REST region includes an amide atom, the full amide group is included.

    Molecule: CH3-C(=O)-NH-CH3
    Starting from the carbonyl C itself, we should get the full amide group (N, C, O)
    plus the C-side methyl + H's. The N-side methyl should not be included.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC"))
    nxg = convert_to_nx(mol)

    amide_n, carbonyl_c, carbonyl_o = _find_amide_atoms(mol)

    result = SingleTopologyREST.expand_rest_region_to_amides({carbonyl_c}, mol, nxg)

    # ALL amide atoms should be included
    assert carbonyl_c in result
    assert carbonyl_o in result
    assert amide_n in result

    # N-side methyl should NOT be included (past the amide boundary)
    methyl_c_n_side = None
    for neighbor in mol.GetAtomWithIdx(amide_n).GetNeighbors():
        idx = neighbor.GetIdx()
        if idx != carbonyl_c and neighbor.GetSymbol() == "C":
            methyl_c_n_side = idx
    assert methyl_c_n_side is not None
    assert methyl_c_n_side not in result
