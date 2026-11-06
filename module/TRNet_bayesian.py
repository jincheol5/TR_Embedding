"""Variational Bayesian encoder layers for the TRNet research extension.

The candidate-message projection is Bayesian; the recurrent gate, source-ID
residual and decoder remain deterministic. This is not a fully Bayesian TRNet.
The diagonal Gaussian posterior and reparameterization follow Bayes by Backprop
(Blundell et al., ICML 2015): https://proceedings.mlr.press/v37/blundell15.html.
We use an isotropic Gaussian prior with an analytic KL, rather than that paper's
scale-mixture prior. Training adds ``kl_weight * kl_loss() / n_train_queries``
to sample-averaged BCE. A nonunit kl_weight is a tempered objective, not the
ordinary ELBO.

Draw once for an entire temporal replay. Never draw separately per graph event:
that would change the sampled recurrent function along the trajectory. The
cache contains epsilon, not a sampled autograd weight graph, so it can be reused
across separate truncated-backpropagation windows without a stale graph.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    """Linear layer with a diagonal Gaussian variational weight posterior.

    Evaluation and training both default to posterior means. Stochasticity is
    enabled explicitly by ``sample_posterior`` and remains coherent until the
    next call to either sampling method. Ephemeral epsilon buffers are excluded
    from checkpoints; the posterior parameters and prior scale are saved.
    """

    def __init__(self,in_features,out_features,bias=False,prior_std=1.0,
                posterior_rho_init=-5.0):
        super().__init__()
        if in_features<1 or out_features<1:
            raise ValueError("BayesianLinear dimensions must be positive")
        if not math.isfinite(prior_std) or prior_std<=0:
            raise ValueError("prior_std must be finite and positive")
        if not math.isfinite(posterior_rho_init):
            raise ValueError("posterior_rho_init must be finite")
        self.in_features=in_features
        self.out_features=out_features
        self.posterior_rho_init=float(posterior_rho_init)
        self.weight_mu=nn.Parameter(torch.empty(out_features,in_features))
        self.weight_rho=nn.Parameter(torch.empty(out_features,in_features))
        if bias:
            self.bias_mu=nn.Parameter(torch.empty(out_features))
            self.bias_rho=nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu",None)
            self.register_parameter("bias_rho",None)
        self.register_buffer("prior_std",torch.tensor(float(prior_std)))
        self.register_buffer("_weight_epsilon",torch.zeros_like(self.weight_mu),persistent=False)
        self.register_buffer("_bias_epsilon",torch.zeros_like(self.bias_mu)
                            if bias else torch.empty(0),persistent=False)
        self._sampled=False
        self.reset_parameters()

    def reset_parameters(self):
        """Match nn.Linear's mean initialization without drawing posterior noise."""
        nn.init.kaiming_uniform_(self.weight_mu,a=math.sqrt(5))
        nn.init.constant_(self.weight_rho,self.posterior_rho_init)
        if self.bias_mu is not None:
            bound=self.in_features**-0.5
            nn.init.uniform_(self.bias_mu,-bound,bound)
            nn.init.constant_(self.bias_rho,self.posterior_rho_init)
        self.use_posterior_mean()

    @staticmethod
    def _sigma(rho):
        # The clamp only protects numerically extreme negative rho values.
        return F.softplus(rho).clamp_min(torch.finfo(rho.dtype).tiny)

    @torch.no_grad()
    def sample_posterior(self,seed=None):
        """Cache one parameter-free draw, preserving global RNG for fixed seeds.

        Fixed-seed noise is generated on CPU then copied to the layer device,
        making the protocol independent of device-specific Generator support.
        Omitting seed consumes the ordinary PyTorch RNG on the current device.
        """
        if seed is None:
            self._weight_epsilon.normal_()
            if self.bias_mu is not None:
                self._bias_epsilon.normal_()
        else:
            generator=torch.Generator(device="cpu").manual_seed(int(seed))
            self._weight_epsilon.copy_(torch.randn(
                self.weight_mu.shape,generator=generator,dtype=self.weight_mu.dtype,
                device="cpu"))
            if self.bias_mu is not None:
                self._bias_epsilon.copy_(torch.randn(
                    self.bias_mu.shape,generator=generator,dtype=self.bias_mu.dtype,
                    device="cpu"))
        self._sampled=True
        return self

    def use_posterior_mean(self):
        self._sampled=False
        return self

    @property
    def is_sampled(self):
        return self._sampled

    def forward(self,x):
        weight=self.weight_mu
        bias=self.bias_mu
        if self._sampled:
            weight=weight+self._sigma(self.weight_rho)*self._weight_epsilon
            if bias is not None:
                bias=bias+self._sigma(self.bias_rho)*self._bias_epsilon
        return F.linear(x,weight,bias)

    def kl_loss(self):
        """Sum KL[q(theta)||N(0,prior_std²)] over every Bayesian scalar."""
        def gaussian_kl(mu,rho):
            sigma=self._sigma(rho)
            return (self.prior_std.log()-sigma.log()
                    +(sigma.square()+mu.square())/(2*self.prior_std.square())-0.5).sum()
        kl=gaussian_kl(self.weight_mu,self.weight_rho)
        if self.bias_mu is not None:
            kl=kl+gaussian_kl(self.bias_mu,self.bias_rho)
        return kl


class BayesianMinimalGRU(nn.Module):
    """minGRU with a variational Bayesian candidate-message projection.

    For a fixed posterior draw the destination recurrence remains affine in its
    previous state. Source input normalization and source-ID residual match the
    deterministic minGRU control, and the unchanged TRNet decoder is external.
    """

    def __init__(self,input_dim,hidden_dim,prior_std=1.0,posterior_rho_init=-5.0):
        super().__init__()
        self.input_norm=nn.LayerNorm(input_dim)
        self.input_proj=BayesianLinear(input_dim,hidden_dim,bias=False,
                                    prior_std=prior_std,posterior_rho_init=posterior_rho_init)
        self.residual=nn.Linear(input_dim,hidden_dim,bias=False)
        self.residual_logit=nn.Parameter(torch.tensor(-2.0))
        self.gate=nn.Linear(input_dim,hidden_dim)
        nn.init.constant_(self.gate.bias,-2.0)

    def sample_posterior(self,seed=None):
        self.input_proj.sample_posterior(seed)
        return self

    def use_posterior_mean(self):
        self.input_proj.use_posterior_mean()
        return self

    @property
    def is_sampled(self):
        return self.input_proj.is_sampled

    def kl_loss(self):
        return self.input_proj.kl_loss()

    def forward(self,x,h_prev,init_src):
        normalized=self.input_norm(x)
        z=torch.sigmoid(self.gate(normalized))
        candidate=(self.input_proj(normalized)
                +torch.sigmoid(self.residual_logit)*self.residual(init_src))
        return (1-z)*h_prev+z*candidate
