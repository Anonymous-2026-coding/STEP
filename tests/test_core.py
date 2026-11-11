"""Meaningful end-to-end invariants for the real neural implementation."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from step import STEP, STEPConfig, build_knn_adjacency
from step.diagnostics import EditControl, edit_rollout
from step._research.transport import unbalanced_relational_sinkhorn


class TestNeuralSTEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        rng=np.random.default_rng(410)
        cls.x=[rng.uniform(.05,.9,(12,12)).astype(np.float32) for _ in range(2)]
        cls.xy=[rng.normal(size=(12,2)).astype(np.float32) for _ in range(2)]
        cls.model=STEP(STEPConfig(latent_dim=5,hidden_dim=16,representation_epochs=3,
            flow_epochs=3,projection_steps=3,integration_step=.25,device="cpu"))
        cls.model.fit_wt(cls.x,cls.xy,[0.,1.])

    def test_neural_transport_flow_present(self):
        self.assertIsInstance(self.model.core.autoencoder_,torch.nn.Module)
        self.assertIsInstance(self.model.core.flow_,torch.nn.Module)
        self.assertGreater(self.model.fit_result.training_pairs,0)
        self.assertGreater(self.model.fit_result.transported_mass_range[0],0)

    def test_instrumented_operator_matches_submitted_core(self):
        x,xy=self.x[0],self.xy[0];graph=build_knn_adjacency(xy,8).numpy()
        rows=np.arange(len(x))==0
        result=edit_rollout(self.model.core,x,graph,2,rows,0.,1.,EditControl("full"))
        original=self.model.core.intervene(x,graph,2,rows,0.,1.,0.)
        keep=np.arange(x.shape[1])!=2
        np.testing.assert_allclose(result["response"][:,keep],original.effect[:,keep],atol=1e-6)
        self.assertAlmostEqual(result["final_raw_residual"],original.projection_residual,places=6)
        self.assertEqual(original.perturbed[0,2],0.)

    def test_checkpoint_roundtrip(self):
        before=self.model.predict_next(1.,2.)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"model.pt";self.model.save(path);loaded=STEP.load(path)
            np.testing.assert_allclose(before,loaded.predict_next(1.,2.),atol=1e-7)

    def test_uot_responds_to_population_mass(self):
        a=np.array([[0.,0.],[1.,1.]],np.float32);b=a+.1
        p,d=unbalanced_relational_sinkhorn(a,b,a,b,source_mass=1.,target_mass=2.)
        q,e=unbalanced_relational_sinkhorn(a,b,a,b,source_mass=1.,target_mass=1.)
        self.assertTrue(np.isfinite(p).all());self.assertGreater(p.sum(),q.sum())

    def test_unknown_source_rejected(self):
        with self.assertRaises(ValueError):self.model.predict_next(.5,1.5)


if __name__=="__main__":unittest.main()
