import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

from tmd.fe.rest.single_topology import SingleTopologyREST
from tmd.graph_utils import convert_to_nx


def get_mol(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=2024)
    return mol


def get_atom_idx_by_map_num(mol: Chem.Mol, map_num: int) -> int:
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    raise ValueError(f"No atom with map number {map_num}")


def expand_from_base(smiles: str, base_idxs: set[int]) -> set[int]:
    """Helper: run the full pipeline (expand_rest_region_in_mol then expand_rest_region_to_nearest_ring)."""
    mol = get_mol(smiles)
    nxg = convert_to_nx(mol)
    cycles = nx.cycle_basis(nxg)
    step1 = SingleTopologyREST.expand_rest_region_in_mol(base_idxs, cycles, mol)
    return SingleTopologyREST.expand_rest_region_to_nearest_ring(step1, base_idxs, mol, nxg, cycles)


class TestExpandRestRegionToNearestRing:
    """Tests for expand_rest_region_to_nearest_ring static method."""

    def test_no_rings_returns_original(self):
        """If the molecule has no rings, return atom_idxs unchanged."""
        mol = get_mol("CCC")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        atom_idxs = {0}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(atom_idxs, atom_idxs, mol, nxg, cycles)
        assert result == atom_idxs

    def test_atom_already_in_ring(self):
        """If starting atoms are already in a ring, no additional expansion from them."""
        mol = get_mol("C1CCCCC1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        single_ring_atom = next(iter(ring_atoms))
        base = {single_ring_atom}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)
        # Ring atoms are not branch atoms, so no BFS occurs
        assert result == base

    def test_branch_atom_reaches_ring(self):
        """An atom on a branch off a ring should expand to include the ring."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        assert ring_atoms.issubset(result)
        assert methyl_c in result

    def test_branch_atom_includes_one_bond_beyond_ring(self):
        """Expansion should include one bond beyond the reached ring for torsion coverage."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1][CH2:2]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        assert ring_atoms.issubset(result)

        for ring_atom in ring_atoms:
            for neighbor in nxg.neighbors(ring_atom):
                assert neighbor in result, (
                    f"Atom {neighbor} (neighbor of ring atom {ring_atom}) should be included "
                    f"as one-bond-beyond for torsion coverage"
                )

    def test_chain_between_branch_and_ring(self):
        """Atoms along the chain between the starting atom and the ring are included."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1][CH2:2][CH2:3]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        ch2_1 = get_atom_idx_by_map_num(mol, 2)
        ch2_2 = get_atom_idx_by_map_num(mol, 3)

        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        assert methyl_c in result
        assert ch2_1 in result
        assert ch2_2 in result

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        assert ring_atoms.issubset(result)

    def test_stops_at_first_ring(self):
        """BFS should stop at the first ring encountered, not continue to further rings."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1ccc(-c2ccccc2)cc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        methyl_neighbors = set(nxg.neighbors(methyl_c))
        attached_ring_idx = None
        for i, cycle in enumerate(cycles):
            if methyl_neighbors & set(cycle):
                attached_ring_idx = i
                break
        assert attached_ring_idx is not None

        attached_ring = set(cycles[attached_ring_idx])
        other_ring_idxs = [i for i in range(len(cycles)) if i != attached_ring_idx]

        assert attached_ring.issubset(result)

        for other_idx in other_ring_idxs:
            other_ring = set(cycles[other_idx])
            one_bond_beyond = set()
            for ra in attached_ring:
                one_bond_beyond.update(set(nxg.neighbors(ra)) & other_ring)
            deep_atoms = other_ring - one_bond_beyond - attached_ring
            for atom in deep_atoms:
                assert atom not in result, (
                    f"Atom {atom} is deep in the second ring and should not be reached"
                )

    def test_fused_rings_entire_system(self):
        """For fused ring systems, the entire fused system should be included."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1cccc2ccccc12"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        assert methyl_c in result

        all_ring_atoms = set()
        for cycle in cycles:
            all_ring_atoms.update(cycle)
        assert all_ring_atoms.issubset(result)

        for ring_atom in all_ring_atoms:
            for neighbor in nxg.neighbors(ring_atom):
                assert neighbor in result

    def test_empty_input(self):
        """Empty atom_idxs should return empty set."""
        mol = get_mol("C1CCCCC1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(set(), set(), mol, nxg, cycles)
        assert result == set()

    def test_multiple_branch_atoms(self):
        """Multiple starting atoms on different branches should each reach their nearest ring."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C1CCC([CH3:2])CC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_1 = get_atom_idx_by_map_num(mol, 1)
        methyl_2 = get_atom_idx_by_map_num(mol, 2)
        base = {methyl_1, methyl_2}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        assert methyl_1 in result
        assert methyl_2 in result
        assert ring_atoms.issubset(result)

    def test_linker_atom_does_not_expand(self):
        """A base atom on a linker between two ring systems should NOT trigger BFS expansion."""
        # phenyl-CH2-phenyl: the CH2 is a linker, not a branch
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc([CH2:1]c2ccccc2)cc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        linker_c = get_atom_idx_by_map_num(mol, 1)
        base = {linker_c}
        # atom_idxs = base (no prior expansion)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        # Linker atom is between two ring systems, so it should NOT be a BFS seed.
        # Result should just be the original set.
        assert result == base

    def test_linker_chain_does_not_expand(self):
        """Base atoms on a chain between two rings should not pull in either ring."""
        # phenyl-CH2-CH2-phenyl: CH2s are linkers
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc([CH2:1][CH2:2]c2ccccc2)cc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        ch2_1 = get_atom_idx_by_map_num(mol, 1)
        ch2_2 = get_atom_idx_by_map_num(mol, 2)
        base = {ch2_1, ch2_2}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        # Neither ring should be pulled in
        assert result == base

    def test_amide_is_not_a_barrier_on_branch(self):
        """On a branch, amide bonds should NOT act as barriers."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C(=O)Nc1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        assert ring_atoms.issubset(result)

    def test_one_bond_beyond_includes_hydrogens(self):
        """One bond beyond the ring should include H atoms on ring carbons."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        for ring_atom in ring_atoms:
            for neighbor in nxg.neighbors(ring_atom):
                if mol.GetAtomWithIdx(neighbor).GetAtomicNum() == 1:
                    assert neighbor in result

    def test_phenyl_substituent_on_branch(self):
        """Branch with a terminal group before the ring — terminal is included in path."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[NH2:1][CH2:2]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        nh2 = get_atom_idx_by_map_num(mol, 1)
        base = {nh2}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        ch2 = get_atom_idx_by_map_num(mol, 2)
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        assert nh2 in result
        assert ch2 in result
        assert ring_atoms.issubset(result)

    def test_fused_tricyclic_system(self):
        """A branch off a tricyclic fused system should include the entire fused system."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1cccc2cc3ccccc3cc12"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        all_ring_atoms = set()
        for cycle in cycles:
            all_ring_atoms.update(cycle)
        assert all_ring_atoms.issubset(result)

    def test_indole_fused_system(self):
        """A branch off indole should include the entire fused bicyclic system."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1ccc2[nH]ccc2c1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        all_ring_atoms = set()
        for cycle in cycles:
            all_ring_atoms.update(cycle)
        assert all_ring_atoms.issubset(result)

    def test_separate_rings_stops_at_first_system(self):
        """Two non-fused rings connected by a chain: only the nearest system is included."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1ccccc1CCc1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        ring_systems = SingleTopologyREST._get_fused_ring_systems(cycles)
        methyl_neighbors = set(nxg.neighbors(methyl_c))
        attached_system = None
        other_system = None
        for system in ring_systems:
            if methyl_neighbors & system:
                attached_system = system
            else:
                other_system = system

        assert attached_system is not None
        assert other_system is not None

        assert attached_system.issubset(result)

        one_bond_beyond = set()
        for atom in attached_system:
            one_bond_beyond.update(nxg.neighbors(atom))
        deep_atoms = other_system - one_bond_beyond
        for atom in deep_atoms:
            assert atom not in result

    def test_drug_like_molecule_linker_atoms_no_expansion(self):
        """In a drug-like molecule, base atoms on linkers between rings should not expand to all rings."""
        # ring1-amide-ring2: base atoms on the amide linker should not pull in either ring
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc([NH:1][C:2](=O)c2ccccc2)cc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        nh = get_atom_idx_by_map_num(mol, 1)
        co = get_atom_idx_by_map_num(mol, 2)
        base = {nh, co}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        # Linker atoms connect two ring systems, so no ring expansion should occur
        all_ring_atoms = set()
        for cycle in cycles:
            all_ring_atoms.update(cycle)
        assert not all_ring_atoms.issubset(result)


    def test_hydrogen_on_ring_does_not_expand(self):
        """H atoms bonded to ring carbons should NOT trigger ring expansion."""
        # Cyclohexane with explicit H — start from an H bonded to a ring carbon
        mol = get_mol("C1CCCCC1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        # Find an H atom bonded to a ring carbon
        h_on_ring = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                nb = atom.GetNeighbors()[0]
                if nb.GetIdx() in ring_atoms:
                    h_on_ring = atom.GetIdx()
                    break
        assert h_on_ring is not None

        base = {h_on_ring}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(base, base, mol, nxg, cycles)

        # The H is a branch atom topologically, but should be filtered out as H
        # No ring expansion should occur
        assert result == base


class TestGetBranchAtoms:
    """Tests for _get_branch_atoms helper."""

    def test_methyl_is_branch(self):
        """A methyl group off a ring is a branch atom."""
        mol = get_mol("CC1CCCCC1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        ring_systems = SingleTopologyREST._get_fused_ring_systems(cycles)

        branch = SingleTopologyREST._get_branch_atoms(nxg, ring_atoms, ring_systems)
        # Atom 0 (methyl C) and its H neighbors should be branch atoms
        assert 0 in branch

    def test_linker_is_not_branch(self):
        """An atom between two ring systems is not a branch atom."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc([CH2:1]c2ccccc2)cc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        ring_systems = SingleTopologyREST._get_fused_ring_systems(cycles)

        linker_c = get_atom_idx_by_map_num(mol, 1)
        branch = SingleTopologyREST._get_branch_atoms(nxg, ring_atoms, ring_systems)
        assert linker_c not in branch

    def test_terminal_off_ring_is_branch(self):
        """A terminal atom (degree 1) off a ring is a branch."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[F:1]c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        ring_systems = SingleTopologyREST._get_fused_ring_systems(cycles)

        f_atom = get_atom_idx_by_map_num(mol, 1)
        branch = SingleTopologyREST._get_branch_atoms(nxg, ring_atoms, ring_systems)
        assert f_atom in branch

    def test_chain_between_rings_is_linker(self):
        """A longer chain between two rings is still a linker."""
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc([CH2:1][CH2:2][CH2:3]c2ccccc2)cc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        ring_systems = SingleTopologyREST._get_fused_ring_systems(cycles)

        branch = SingleTopologyREST._get_branch_atoms(nxg, ring_atoms, ring_systems)
        for map_num in [1, 2, 3]:
            atom = get_atom_idx_by_map_num(mol, map_num)
            assert atom not in branch


class TestExpandRestRegionInMol:
    """Tests for expand_rest_region_in_mol to ensure terminal expansion still works."""

    def test_terminal_atoms_added(self):
        """Terminal (degree-1) atoms adjacent to REST atoms should be included."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        result = SingleTopologyREST.expand_rest_region_in_mol(ring_atoms, cycles, mol)

        for ring_atom in ring_atoms:
            for neighbor_atom in mol.GetAtomWithIdx(ring_atom).GetNeighbors():
                if len(neighbor_atom.GetNeighbors()) == 1:
                    assert neighbor_atom.GetIdx() in result

    def test_complete_ring_included(self):
        """If any atom of a ring is in the input, the whole ring should be included."""
        mol = get_mol("C1CCCCC1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        single_atom = next(iter(ring_atoms))
        result = SingleTopologyREST.expand_rest_region_in_mol({single_atom}, cycles, mol)
        assert ring_atoms.issubset(result)


class TestPipelineIntegration:
    """Test that expand_rest_region_in_mol followed by expand_rest_region_to_nearest_ring
    works correctly as a pipeline."""

    def test_branch_atom_pipeline(self):
        """Full pipeline: terminal expansion then ring expansion from a branch atom."""
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1][CH2:2]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        base = {methyl_c}

        step1 = SingleTopologyREST.expand_rest_region_in_mol(base, cycles, mol)
        step2 = SingleTopologyREST.expand_rest_region_to_nearest_ring(step1, base, mol, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        assert ring_atoms.issubset(step2)
        assert methyl_c in step2
        assert get_atom_idx_by_map_num(mol, 2) in step2

    def test_ring_atom_pipeline_no_spurious_expansion(self):
        """Starting from a ring atom, the pipeline should not expand to distant rings."""
        mol = get_mol("c1ccc(-c2ccccc2)cc1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        ring_0 = set(cycles[0])
        ring_1 = set(cycles[1])
        shared_or_adjacent = ring_1.copy()
        for a in ring_1:
            shared_or_adjacent.update(nxg.neighbors(a))
        isolated_atoms = ring_0 - shared_or_adjacent

        if isolated_atoms:
            start_atom = next(iter(isolated_atoms))
            base = {start_atom}

            step1 = SingleTopologyREST.expand_rest_region_in_mol(base, cycles, mol)
            assert ring_0.issubset(step1)

            step2 = SingleTopologyREST.expand_rest_region_to_nearest_ring(step1, base, mol, nxg, cycles)

            one_bond_beyond = set()
            for ra in ring_0:
                one_bond_beyond.update(set(nxg.neighbors(ra)) & ring_1)
            deep_atoms = ring_1 - one_bond_beyond - ring_0
            for atom in deep_atoms:
                assert atom not in step2

    def test_linker_base_atoms_dont_expand_pipeline(self):
        """Base atoms on linkers between rings: pipeline should not pull in distant rings."""
        # ring1-CH2-CH2-ring2, base atoms are the CH2s
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc([CH2:1][CH2:2]c2ccccc2)cc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        ch2_1 = get_atom_idx_by_map_num(mol, 1)
        ch2_2 = get_atom_idx_by_map_num(mol, 2)
        base = {ch2_1, ch2_2}

        step1 = SingleTopologyREST.expand_rest_region_in_mol(base, cycles, mol)
        step2 = SingleTopologyREST.expand_rest_region_to_nearest_ring(step1, base, mol, nxg, cycles)

        # Neither ring should be fully included
        all_ring_atoms = set()
        for cycle in cycles:
            all_ring_atoms.update(cycle)
        assert not all_ring_atoms.issubset(step2)
