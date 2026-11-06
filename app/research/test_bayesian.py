"""Bayesian-layer algebra, coherent sampling, and recurrence gradient checks."""
import copy
import unittest

import torch
from torch import nn
from torch.distributions import Normal,kl_divergence

from module.TRNet_bayesian import BayesianLinear,BayesianMinimalGRU
from module.TRNet_module import MinimalGRU


class BayesianTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def test_mean_matches_linear_and_minimal_gru_control(self):
        torch.manual_seed(31)
        linear=nn.Linear(5,3,bias=True)
        torch.manual_seed(31)
        bayesian=BayesianLinear(5,3,bias=True)
        x=torch.randn(4,5)
        torch.testing.assert_close(bayesian(x),linear(x),rtol=0,atol=0)
        self.assertFalse(bayesian.is_sampled)
        torch.manual_seed(37)
        control=MinimalGRU(5,5)
        torch.manual_seed(37)
        recurrent=BayesianMinimalGRU(5,5)
        h,identity=torch.randn(4,5),torch.randn(4,5)
        torch.testing.assert_close(control(x,h,identity),recurrent(x,h,identity),rtol=0,atol=0)

    def test_coherent_draw_reproduces_without_touching_global_rng(self):
        layer=BayesianLinear(4,3,bias=True)
        x=torch.randn(7,4)
        mean=layer(x).detach()
        rng=torch.get_rng_state().clone()
        layer.sample_posterior(seed=53)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        first=layer(x).detach()
        torch.testing.assert_close(first,layer(x),rtol=0,atol=0)
        layer.sample_posterior(seed=53)
        torch.testing.assert_close(first,layer(x),rtol=0,atol=0)
        layer.sample_posterior(seed=59)
        self.assertFalse(torch.equal(first,layer(x)))
        layer.eval()
        self.assertTrue(layer.is_sampled)  # Mode changes do not silently alter a replay draw.
        layer.use_posterior_mean()
        torch.testing.assert_close(mean,layer(x),rtol=0,atol=0)

    def test_analytic_kl_matches_torch_distributions(self):
        for bias in (False,True):
            layer=BayesianLinear(3,2,bias=bias,prior_std=0.1).double()
            expected=kl_divergence(Normal(layer.weight_mu,nn.functional.softplus(layer.weight_rho)),
                                   Normal(torch.zeros_like(layer.weight_mu),layer.prior_std)).sum()
            if bias:
                expected=expected+kl_divergence(
                    Normal(layer.bias_mu,nn.functional.softplus(layer.bias_rho)),
                    Normal(torch.zeros_like(layer.bias_mu),layer.prior_std)).sum()
            torch.testing.assert_close(layer.kl_loss(),expected,rtol=1e-12,atol=1e-12)
        with torch.no_grad():
            layer.weight_mu.zero_()
            layer.weight_rho.fill_(torch.log(torch.expm1(layer.prior_std)))
            layer.bias_mu.zero_()
            layer.bias_rho.fill_(torch.log(torch.expm1(layer.prior_std)))
        torch.testing.assert_close(layer.kl_loss(),torch.zeros((),dtype=torch.float64),atol=1e-12,rtol=0)

    def test_gradients_survive_separate_bptt_windows_and_same_noise(self):
        cell=BayesianMinimalGRU(5,5,prior_std=0.1)
        cell.sample_posterior(seed=67)
        optimizer=torch.optim.Adam(cell.parameters(),lr=0.001)
        h=torch.zeros(3,5)
        epsilon=cell.input_proj._weight_epsilon.clone()
        for _ in range(2):
            optimizer.zero_grad()
            for _ in range(3):
                x,identity=torch.randn(3,5),torch.randn(3,5)
                h=cell(x,h,identity)
            loss=h.square().mean()+cell.kl_loss()/1000
            loss.backward()
            for name,parameter in cell.named_parameters():
                self.assertIsNotNone(parameter.grad,name)
                self.assertTrue(torch.isfinite(parameter.grad).all(),name)
            self.assertGreater(cell.input_proj.weight_rho.grad.abs().sum().item(),0)
            optimizer.step()
            h=h.detach()
        self.assertTrue(torch.equal(epsilon,cell.input_proj._weight_epsilon))

    def test_posterior_spread_changes_predictive_variance(self):
        small=BayesianLinear(3,2)
        large=copy.deepcopy(small)
        with torch.no_grad():
            large.weight_rho.fill_(-2.0)
        x=torch.tensor([[1.0,-0.5,2.0]])
        outputs_small,outputs_large=[],[]
        for seed in range(80):
            small.sample_posterior(seed)
            large.sample_posterior(seed)
            outputs_small.append(small(x).detach())
            outputs_large.append(large(x).detach())
        variance_small=torch.stack(outputs_small).var(0)
        variance_large=torch.stack(outputs_large).var(0)
        ratio=(nn.functional.softplus(large.weight_rho[0,0])
               /nn.functional.softplus(small.weight_rho[0,0])).square()
        torch.testing.assert_close(variance_large,variance_small*ratio,rtol=1e-5,atol=1e-7)

    def test_checkpoints_store_posterior_but_not_ephemeral_draw(self):
        layer=BayesianLinear(3,2,prior_std=0.1)
        layer.sample_posterior(71)
        state=layer.state_dict()
        self.assertIn("prior_std",state)
        self.assertFalse(any("epsilon" in name for name in state))
        restored=BayesianLinear(3,2)
        restored.load_state_dict(state)
        self.assertFalse(restored.is_sampled)
        layer.use_posterior_mean()
        x=torch.randn(4,3)
        torch.testing.assert_close(layer(x),restored(x),rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
