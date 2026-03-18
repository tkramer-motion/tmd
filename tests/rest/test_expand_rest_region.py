import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

from tmd.fe.rest.single_topology import SingleTopologyREST
from tmd.graph_utils import convert_to_nx


def get_mol(smiles: str) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=2024)
    return mol


def get_ring_expansion(smiles: str, atom_idxs: set[int]) -> set[int]:
    """Helper: run expand_rest_region_to_nearest_ring on a molecule."""
    mol = get_mol(smiles)
    nxg = convert_to_nx(mol)
    cycles = nx.cycle_basis(nxg)
    return SingleTopologyREST.expand_rest_region_to_nearest_ring(atom_idxs, nxg, cycles)


def get_heavy_atom_idxs(mol: Chem.Mol) -> set[int]:
    return {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1}


def get_atom_idx_by_map_num(mol: Chem.Mol, map_num: int) -> int:
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_num:
            return atom.GetIdx()
    raise ValueError(f"No atom with map number {map_num}")


def get_ring_atom_idxs(mol: Chem.Mol) -> set[int]:
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    return ring_atoms


class TestExpandRestRegionToNearestRing:
    """Tests for expand_rest_region_to_nearest_ring static method."""

    def test_no_rings_returns_original(self):
        """If the molecule has no rings, return atom_idxs unchanged."""
        # propane: C-C-C
        mol = get_mol("CCC")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        atom_idxs = {0}
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(atom_idxs, nxg, cycles)
        assert result == atom_idxs

    def test_atom_already_in_ring(self):
        """If starting atoms are already in a ring, no additional expansion from them."""
        # cyclohexane
        mol = get_mol("C1CCCCC1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        # Start from a single ring atom — should not expand beyond the ring + one bond beyond
        single_ring_atom = next(iter(ring_atoms))
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({single_ring_atom}, nxg, cycles)
        # Should just be the input since it's already a ring atom (and non-ring atoms don't BFS)
        assert result == {single_ring_atom}

    def test_branch_atom_reaches_ring(self):
        """An atom on a branch off a ring should expand to include the ring."""
        # methylcyclohexane: methyl group hanging off cyclohexane
        # Use mapped SMILES to identify the methyl carbon
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        # Should include the methyl carbon + entire ring
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        assert ring_atoms.issubset(result)
        assert methyl_c in result

    def test_branch_atom_includes_one_bond_beyond_ring(self):
        """Expansion should include one bond beyond the reached ring for torsion coverage."""
        # 1-ethylcyclohexane: ethyl group on one side, ring, then check the other side
        # [CH3:1][CH2:2]C1CCCCC1 — start from the terminal methyl
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1][CH2:2]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        # All ring atoms should be included
        assert ring_atoms.issubset(result)

        # One bond beyond: neighbors of ring atoms not on the path should be included
        # This includes H atoms on ring carbons and any substituents on the far side
        for ring_atom in ring_atoms:
            for neighbor in nxg.neighbors(ring_atom):
                assert neighbor in result, (
                    f"Atom {neighbor} (neighbor of ring atom {ring_atom}) should be included "
                    f"as one-bond-beyond for torsion coverage"
                )

    def test_chain_between_branch_and_ring(self):
        """Atoms along the chain between the starting atom and the ring are included."""
        # [CH3:1][CH2:2][CH2:3]C1CCCCC1 — propyl chain off cyclohexane
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1][CH2:2][CH2:3]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        ch2_1 = get_atom_idx_by_map_num(mol, 2)
        ch2_2 = get_atom_idx_by_map_num(mol, 3)

        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        # All atoms along the chain should be included
        assert methyl_c in result
        assert ch2_1 in result
        assert ch2_2 in result

        # Ring should be included
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)
        assert ring_atoms.issubset(result)

    def test_stops_at_first_ring(self):
        """BFS should stop at the first ring encountered, not continue to further rings."""
        # biphenyl with a methyl on the bridge: two rings connected by a bond
        # Start from methyl on one ring, should reach that ring but NOT the second ring
        # Use: [CH3:1]c1ccc(-c2ccccc2)cc1
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1ccc(-c2ccccc2)cc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        # Identify which ring the methyl is attached to
        methyl_neighbors = set(nxg.neighbors(methyl_c))
        attached_ring_idx = None
        for i, cycle in enumerate(cycles):
            if methyl_neighbors & set(cycle):
                attached_ring_idx = i
                break
        assert attached_ring_idx is not None

        attached_ring = set(cycles[attached_ring_idx])
        other_ring_idxs = [i for i in range(len(cycles)) if i != attached_ring_idx]

        # The attached ring should be included
        assert attached_ring.issubset(result)

        # The other ring should NOT be fully included (only atoms one bond beyond may be)
        # Specifically, atoms of the other ring that are NOT neighbors of the attached ring
        # should not be in the result
        for other_idx in other_ring_idxs:
            other_ring = set(cycles[other_idx])
            # Atoms in the other ring that are neighbors of the attached ring are "one bond beyond"
            one_bond_beyond = set()
            for ra in attached_ring:
                one_bond_beyond.update(set(nxg.neighbors(ra)) & other_ring)
            # Atoms deeper in the other ring should NOT be included
            deep_atoms = other_ring - one_bond_beyond - attached_ring
            for atom in deep_atoms:
                assert atom not in result, (
                    f"Atom {atom} is deep in the second ring and should not be reached"
                )

    def test_fused_rings_entire_system(self):
        """For fused ring systems, the entire fused system should be included."""
        # naphthalene with methyl: [CH3:1]c1cccc2ccccc12
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1cccc2ccccc12"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        # The methyl should be in the result
        assert methyl_c in result

        # ALL ring atoms (both fused rings) should be included
        all_ring_atoms = set()
        for cycle in cycles:
            all_ring_atoms.update(cycle)
        assert all_ring_atoms.issubset(result)

        # One bond beyond the fused system should also be included
        for ring_atom in all_ring_atoms:
            for neighbor in nxg.neighbors(ring_atom):
                assert neighbor in result

    def test_empty_input(self):
        """Empty atom_idxs should return empty set."""
        mol = get_mol("C1CCCCC1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring(set(), nxg, cycles)
        assert result == set()

    def test_multiple_branch_atoms(self):
        """Multiple starting atoms on different branches should each reach their nearest ring."""
        # Two methyls on cyclohexane: [CH3:1]C1CCC([CH3:2])CC1
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C1CCC([CH3:2])CC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_1 = get_atom_idx_by_map_num(mol, 1)
        methyl_2 = get_atom_idx_by_map_num(mol, 2)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_1, methyl_2}, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        assert methyl_1 in result
        assert methyl_2 in result
        assert ring_atoms.issubset(result)

    def test_branch_between_two_rings(self):
        """A branch atom between two separate rings should reach the nearest one(s)."""
        # [CH2:1](C1CCCCC1)C1CCCCC1 — methylene bridging two cyclohexanes
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH2:1](C1CCCCC1)C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        bridge_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({bridge_c}, nxg, cycles)

        # Both rings are equidistant (1 bond away), so both should be included
        for cycle in cycles:
            assert set(cycle).issubset(result)

    def test_amide_is_not_a_barrier(self):
        """Unlike the old amide expansion, amide bonds should NOT act as barriers."""
        # acetanilide: CC(=O)Nc1ccccc1 — methyl -> amide -> phenyl
        # Start from methyl, should pass through amide and reach the ring
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C(=O)Nc1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        # Should reach the phenyl ring through the amide
        assert ring_atoms.issubset(result)

    def test_one_bond_beyond_includes_hydrogens(self):
        """One bond beyond the ring should include H atoms on ring carbons."""
        # methylcyclohexane — start from methyl
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        # All H atoms bonded to ring atoms should be in the result (one bond beyond)
        for ring_atom in ring_atoms:
            for neighbor in nxg.neighbors(ring_atom):
                if mol.GetAtomWithIdx(neighbor).GetAtomicNum() == 1:
                    assert neighbor in result

    def test_phenyl_substituent_on_branch(self):
        """Branch with a terminal group before the ring — terminal is included in path."""
        # 4-(2-aminoethyl)cyclohexane-like: [NH2:1][CH2:2]C1CCCCC1
        mol = Chem.AddHs(Chem.MolFromSmiles("[NH2:1][CH2:2]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        nh2 = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({nh2}, nxg, cycles)

        ch2 = get_atom_idx_by_map_num(mol, 2)
        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        assert nh2 in result
        assert ch2 in result
        assert ring_atoms.issubset(result)

    def test_fused_tricyclic_system(self):
        """A branch off a tricyclic fused system should include the entire fused system."""
        # methyl on anthracene: [CH3:1]c1cccc2cc3ccccc3cc12
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1cccc2cc3ccccc3cc12"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        # All three fused rings should be included
        all_ring_atoms = set()
        for cycle in cycles:
            all_ring_atoms.update(cycle)
        assert all_ring_atoms.issubset(result)

    def test_indole_fused_system(self):
        """A branch off indole should include the entire fused bicyclic system."""
        # methyl on indole: [CH3:1]c1ccc2[nH]ccc2c1
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1ccc2[nH]ccc2c1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        all_ring_atoms = set()
        for cycle in cycles:
            all_ring_atoms.update(cycle)
        assert all_ring_atoms.issubset(result)

    def test_separate_rings_stops_at_first_system(self):
        """Two non-fused rings connected by a chain: only the nearest system is included."""
        # [CH3:1]c1ccccc1CCc1ccccc1 — methyl on phenyl, chain to second phenyl
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]c1ccccc1CCc1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)
        result = SingleTopologyREST.expand_rest_region_to_nearest_ring({methyl_c}, nxg, cycles)

        # Identify the ring system attached to the methyl
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

        # The attached system should be fully included
        assert attached_system.issubset(result)

        # The other system should NOT be fully included
        # Only atoms one bond beyond the attached system may appear
        one_bond_beyond = set()
        for atom in attached_system:
            one_bond_beyond.update(nxg.neighbors(atom))
        deep_atoms = other_system - one_bond_beyond
        for atom in deep_atoms:
            assert atom not in result


class TestExpandRestRegionInMol:
    """Tests for expand_rest_region_in_mol to ensure terminal expansion still works."""

    def test_terminal_atoms_added(self):
        """Terminal (degree-1) atoms adjacent to REST atoms should be included."""
        # methylcyclohexane
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        # expand_rest_region_in_mol with ring atoms as input
        result = SingleTopologyREST.expand_rest_region_in_mol(ring_atoms, cycles, mol)

        # H atoms on ring carbons (degree 1) should be included
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
        # ethylcyclohexane: start from terminal methyl
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3:1][CH2:2]C1CCCCC1"))
        AllChem.EmbedMolecule(mol, randomSeed=2024)
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        methyl_c = get_atom_idx_by_map_num(mol, 1)

        # Step 1: ring + terminal expansion
        step1 = SingleTopologyREST.expand_rest_region_in_mol({methyl_c}, cycles, mol)

        # Step 2: nearest ring expansion
        step2 = SingleTopologyREST.expand_rest_region_to_nearest_ring(step1, nxg, cycles)

        ring_atoms = set()
        for cycle in cycles:
            ring_atoms.update(cycle)

        # After pipeline, ring should be included
        assert ring_atoms.issubset(step2)
        # Methyl and CH2 should be included
        assert methyl_c in step2
        assert get_atom_idx_by_map_num(mol, 2) in step2

    def test_ring_atom_pipeline_no_spurious_expansion(self):
        """Starting from a ring atom, the pipeline should not expand to distant rings."""
        # biphenyl: c1ccc(-c2ccccc2)cc1
        mol = get_mol("c1ccc(-c2ccccc2)cc1")
        nxg = convert_to_nx(mol)
        cycles = nx.cycle_basis(nxg)

        # Pick an atom from one ring
        ring_0 = set(cycles[0])
        # Pick an atom that's NOT shared with the other ring and NOT adjacent to it
        ring_1 = set(cycles[1])
        shared_or_adjacent = ring_1.copy()
        for a in ring_1:
            shared_or_adjacent.update(nxg.neighbors(a))
        isolated_atoms = ring_0 - shared_or_adjacent

        if isolated_atoms:
            start_atom = next(iter(isolated_atoms))

            # Step 1: ring + terminal expansion — should include ring_0
            step1 = SingleTopologyREST.expand_rest_region_in_mol({start_atom}, cycles, mol)
            assert ring_0.issubset(step1)

            # Step 2: nearest ring expansion — ring atoms don't BFS, so ring_1 should not be fully included
            step2 = SingleTopologyREST.expand_rest_region_to_nearest_ring(step1, nxg, cycles)

            # Atoms deep in ring_1 (not one-bond-beyond ring_0) should not be included
            one_bond_beyond = set()
            for ra in ring_0:
                one_bond_beyond.update(set(nxg.neighbors(ra)) & ring_1)
            deep_atoms = ring_1 - one_bond_beyond - ring_0
            for atom in deep_atoms:
                assert atom not in step2
