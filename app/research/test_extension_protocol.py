"""Toy-graph checks for Bayesian training and coherent posterior replay.

No stored benchmark validation/test data are used by these tests.
"""
import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score

from app.research.run import build_model
from model_train.TRNet_train import TRNet_Trainer


def events(src,dst,timestamp):
    return tuple(torch.tensor(value,dtype=torch.long) for value in (src,dst,timestamp))


def queries(src,dst,timestamp,label):
    return {"src":torch.tensor(src,dtype=torch.long),
            "dst":torch.tensor(dst,dtype=torch.long),
            "query_t":torch.full((len(src),),timestamp,dtype=torch.long),
            "label":torch.tensor(label,dtype=torch.float32)}


class ExtensionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(127)
        # The timestamp-2 chain crosses a batch boundary. Its second hop must
        # not expose reachability acquired by the first hop at the same time.
        self.train=[events([1],[2],[1]),events([2],[3],[2]),events([3],[4],[2])]
        self.train_queries=[queries([1,3],[2,1],1,[1,0]),
                            queries([1,4],[3,1],2,[1,0]),
                            queries([3,1],[4,4],2,[1,0])]
        self.val=[events([2,4],[4,1],[3,3]),events([4],[3],[4])]
        self.val_queries=[queries([1,4],[4,3],3,[1,0]),
                          queries([4,2,3,1],[3,1,1,2],4,[1,0,1,1])]

    def model(self,encoder="bayesian_mingru",rho=-1.0):
        config={"encoder_type":encoder,"decoder_input":"concat","mem_dim":8,
                "latent_dim":8,"event_backend":"wave",
                "bayes_prior_std":0.5,"bayes_posterior_rho_init":rho}
        model=build_model(config)
        model.set_module(4)
        return model

    def manual_draws(self,model,count,seed):
        probabilities=[]
        logits_all=[]
        with torch.no_grad():
            for draw in range(count):
                model.encoder.sample_posterior(seed+draw)
                model.eval()
                model.reset_memory()
                for batch in self.train:
                    model.update_mem(*batch)
                logits=[]
                for batch,query in zip(self.val,self.val_queries):
                    model.update_mem(*batch)
                    logits.append(model(query["src"],query["dst"]))
                logits=torch.cat(logits)
                logits_all.append(logits.double())
                probabilities.append(torch.sigmoid(logits).double())
        model.encoder.use_posterior_mean()
        return torch.stack(probabilities),torch.stack(logits_all)

    def test_return_scores_preserves_original_metric_contract(self):
        model=self.model("mingru")
        plain=TRNet_Trainer.evaluate_research(
            model,self.val,self.val_queries,prefix_batches=(self.train,))
        detailed,logits,labels=TRNet_Trainer.evaluate_research(
            model,self.val,self.val_queries,prefix_batches=(self.train,),return_scores=True)
        self.assertEqual(plain,detailed)
        self.assertEqual(set(plain),{"loss","acc","ap","auc","n"})
        self.assertEqual(logits.shape,(6,))
        self.assertFalse(logits.requires_grad)
        self.assertEqual(logits.device.type,"cpu")
        self.assertTrue(torch.equal(labels,torch.cat([q["label"] for q in self.val_queries])))
        expected=nn.functional.binary_cross_entropy_with_logits(logits,labels)
        self.assertAlmostEqual(plain["loss"],expected.item(),places=7)
        self.assertEqual(plain["acc"],float(((logits>=0)==labels).double().mean()))
        self.assertEqual(plain["ap"],average_precision_score(labels.numpy(),logits.numpy()))
        self.assertEqual(plain["auc"],roc_auc_score(labels.numpy(),logits.numpy()))

    def test_mc_metrics_equal_independent_full_prefix_probability_average(self):
        model=self.model(rho=-0.7)
        # Increase readout contrast enough to distinguish E[sigmoid(logit)]
        # from sigmoid(E[logit]) at ordinary float32 precision.
        with torch.no_grad():
            model.decoder.mlp[-1].weight.mul_(12)
            model.decoder.mlp[-1].bias.fill_(0.7)
        draws,logits=self.manual_draws(copy.deepcopy(model),4,601)
        probability=draws.mean(0)
        labels=torch.cat([q["label"] for q in self.val_queries]).double()
        self.assertGreater((probability-torch.sigmoid(logits.mean(0))).abs().max().item(),1e-7)
        actual=TRNet_Trainer.evaluate_bayesian_research(
            model,self.val,self.val_queries,prefix_batches=(self.train,),
            mc_samples=4,mc_seed=601)
        expected={"loss":float(nn.functional.binary_cross_entropy(probability,labels)),
                  "acc":float(((probability>=0.5)==labels).double().mean()),
                  "ap":float(average_precision_score(labels.numpy(),probability.numpy())),
                  "auc":float(roc_auc_score(labels.numpy(),probability.numpy())),
                  "mc_probability_std_mean":float(draws.std(0,unbiased=False).mean())}
        for metric,value in expected.items():
            self.assertAlmostEqual(actual[metric],value,places=13,msg=metric)
        self.assertEqual(actual["n"],len(labels))
        self.assertEqual(actual["mc_samples"],4)
        self.assertEqual(actual["mc_seed"],601)
        self.assertFalse(model.encoder.is_sampled)

    def test_each_mc_draw_resets_history_and_reuses_one_sample(self):
        model=self.model()
        original_reset=model.reset_memory
        original_update=model.update_mem
        sample_noise=[]
        active_noise=None
        updates=[]

        def reset():
            nonlocal active_noise
            self.assertTrue(model.encoder.is_sampled)
            active_noise=model.encoder.input_proj._weight_epsilon.clone()
            sample_noise.append(active_noise)
            original_reset()
            torch.testing.assert_close(model.module.cumul_mem,model.module.init_mem,rtol=0,atol=0)
            self.assertTrue(torch.equal(model.module.last_t,torch.zeros_like(model.module.last_t)))

        def update(src,dst,event_t):
            self.assertTrue(model.encoder.is_sampled)
            self.assertTrue(torch.equal(active_noise,model.encoder.input_proj._weight_epsilon))
            updates.append((len(sample_noise),event_t.tolist()))
            return original_update(src,dst,event_t)

        rng=torch.get_rng_state().clone()
        with patch.object(model,"reset_memory",side_effect=reset), \
             patch.object(model,"update_mem",side_effect=update):
            first=TRNet_Trainer.evaluate_bayesian_research(
                model,self.val,self.val_queries,prefix_batches=(self.train,),
                mc_samples=3,mc_seed=709)
        self.assertEqual(len(sample_noise),3)
        self.assertEqual(len(updates),3*(len(self.train)+len(self.val)))
        self.assertFalse(torch.equal(sample_noise[0],sample_noise[1]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        # Deliberately contaminate every mutable vector state. The next MC call
        # must be exactly reproducible because each draw resets the full prefix.
        model.module.cumul_mem.fill_(91)
        model.module.prev_mem.fill_(-47)
        model.module.last_t.fill_(999)
        second=TRNet_Trainer.evaluate_bayesian_research(
            model,self.val,self.val_queries,prefix_batches=(self.train,),
            mc_samples=3,mc_seed=709)
        self.assertEqual(first,second)

    def test_two_epoch_training_uses_bptt_windows_and_scaled_kl(self):
        model=self.model(rho=-4.0)
        rho=model.encoder.input_proj.weight_rho
        before_rho=rho.detach().clone()
        before_mu=model.encoder.input_proj.weight_mu.detach().clone()
        gradients=[]
        hook=rho.register_hook(lambda gradient:gradients.append(gradient.detach().clone()))
        original_kl=model.encoder.kl_loss
        kl_values=[]

        def record_kl():
            value=original_kl()
            kl_values.append(value.detach().item())
            return value

        try:
            with patch.object(model.encoder,"kl_loss",side_effect=record_kl):
                result=TRNet_Trainer.fit_research(
                    model,self.train,self.val,self.val_queries,
                    lambda epoch:self.train_queries,epochs=2,patience=2,bptt_batches=2,
                    lr=0.001,kl_weight=0.01,mc_samples=2,bayesian_seed=809)
        finally:
            hook.remove()
        self.assertEqual(result["epochs_run"],2)
        self.assertEqual(len(gradients),4)  # Full 2-batch window + partial window per epoch.
        self.assertEqual(len(kl_values),6)  # Two objective calls + one diagnostic per epoch.
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(all(gradient.abs().sum().item()>0 for gradient in gradients))
        self.assertFalse(torch.equal(before_rho,rho))
        self.assertFalse(torch.equal(before_mu,model.encoder.input_proj.weight_mu))
        for epoch,entry in enumerate(result["history"]):
            expected=0.01*sum(kl_values[3*epoch:3*epoch+2])/2/6
            self.assertAlmostEqual(entry["scaled_kl_per_step"],expected,places=7)
            self.assertEqual(entry["train_query_count"],6)
            self.assertEqual(entry["kl_weight"],0.01)
            self.assertTrue(math.isfinite(entry["train_loss"]))
            for metric in ("loss","acc","ap","auc","mc_probability_std_mean"):
                self.assertTrue(math.isfinite(entry["validation"][metric]))
        self.assertIsNone(model.module.cumul_mem.grad_fn)
        self.assertFalse(model.encoder.is_sampled)
        rebuilt=TRNet_Trainer.evaluate_bayesian_research(
            model,self.val,self.val_queries,prefix_batches=(self.train,),
            mc_samples=2,mc_seed=809+500000)
        self.assertEqual(rebuilt,result["validation"])

    def test_evaluation_failure_clears_sampled_mode(self):
        model=self.model()
        invalid=copy.deepcopy(self.val_queries)
        invalid[0]["query_t"].fill_(987)
        with self.assertRaisesRegex(ValueError,"timestamp"):
            TRNet_Trainer.evaluate_bayesian_research(
                model,self.val,invalid,prefix_batches=(self.train,),mc_samples=2)
        self.assertFalse(model.encoder.is_sampled)
        with self.assertRaisesRegex(ValueError,"positive"):
            TRNet_Trainer.evaluate_bayesian_research(model,self.val,self.val_queries,mc_samples=0)


if __name__=="__main__":
    unittest.main()
