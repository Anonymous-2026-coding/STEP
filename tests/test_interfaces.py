"""Round trips and semantic checks for user-facing inputs and stationary edits."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from step import STEP, StaticSTEP, StationaryConfig, build_knn_adjacency
from step.io import load_snapshots, save_snapshots


class TestSnapshotIO(unittest.TestCase):
    def test_variable_population_and_expression_precision(self):
        rng = np.random.default_rng(14)
        expression = [rng.random((n, 3)) for n in (6, 9, 7)]
        coordinates = [rng.normal(size=(len(x), 2)) for x in expression]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.npz"
            save_snapshots(path, expression, coordinates, [0, 2, 5], ["a", "b", "c"])
            loaded = load_snapshots(path)
        self.assertEqual([len(x) for x in loaded.expression], [6, 9, 7])
        self.assertEqual(loaded.genes, ["a", "b", "c"])
        for expected, actual in zip(expression, loaded.expression):
            np.testing.assert_array_equal(expected, actual)


class TestStaticInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng = np.random.default_rng(24)
        cls.x = rng.uniform(.05, .9, (12, 6)).astype(np.float32)
        cls.xy = rng.normal(size=(12, 2)).astype(np.float32)
        cls.graph = build_knn_adjacency(cls.xy, 3).numpy()
        cls.model = StaticSTEP(StationaryConfig(
            latent_dim=3, hidden_dim=12, representation_epochs=3, flow_epochs=3,
            neighbors=3, integration_step=.25, projection_steps=3, seed=24, device="cpu",
        )).fit(cls.x, cls.xy, adjacency=cls.graph, genes=[f"g{i}" for i in range(6)])

    def test_graph_only_query_checkpoint_and_gene_identity(self):
        rows = np.arange(len(self.x)) == 0
        expected = self.model.perturb(self.x, None, "g2", rows, adjacency=self.graph)
        self.assertEqual(expected.perturbed[0, 2], 0.)
        self.assertGreaterEqual(expected.projection_residual, 0.)
        np.testing.assert_allclose(expected.effect, expected.perturbed - expected.baseline, atol=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.model.save(path)
            loaded = STEP.load(path, device="cpu")
            self.assertIsInstance(loaded, StaticSTEP)
            actual = loaded.perturb(self.x, None, "g2", rows, adjacency=self.graph)
        np.testing.assert_allclose(actual.effect, expected.effect, atol=1e-7)
        self.assertEqual(loaded.genes, self.model.genes)

    def test_grn_regulator_columns_and_self_edges(self):
        matrix = self.model.extract_grn(self.x, None, 0., regulators=["g4", "g1"],
                                        adjacency=self.graph, remove_self=False)
        reference = self.model.core.dynamic_grn(self.x, self.graph, 0.)
        np.testing.assert_allclose(matrix, reference[:, [4, 1]], atol=1e-7)
        masked = self.model.extract_grn(self.x, None, 0., regulators=["g4", "g1"],
                                        adjacency=self.graph)
        reference = matrix.copy()
        reference[4, 0] = reference[1, 1] = 0.
        np.testing.assert_array_equal(masked, reference)

    def test_ambiguous_edit_mask_rejected(self):
        with self.assertRaises(ValueError):
            self.model.perturb(self.x, self.xy, "g2", np.zeros(len(self.x), dtype=int))
        with self.assertRaises(ValueError):
            self.model.perturb(self.x, self.xy, "unknown", np.arange(len(self.x)) == 0)


if __name__ == "__main__":
    unittest.main()
