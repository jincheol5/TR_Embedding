"""Correctness checks for temporal replay and research encoder lifecycles.

Run: python -m unittest app.research.test_encoders
"""
import copy
import unittest

import torch

from model.TRNet import TRNet


def legacy_update(model,src,dst,event_t):
    """Original event-by-event implementation, retained as a parity oracle."""
    for i in range(src.numel()):
        s,d=src[i].reshape(1),dst[i].reshape(1)
        t=event_t[i]
        x=model.module.prev_mem[s] if model.module.last_t[s].item()==t.item() else model.module.cumul_mem[s]
        old=model.module.cumul_mem[d]
        if model.module.last_t[d].item()<t.item():
            model.module.update_prev_mem(d,old)
        new=model.encoder(x=x,h_prev=old,init_src=model.module.init_mem[s])
        model.module.update_cumul_mem(d,new)
        model.module.update_last_t(d,t.reshape(1))


class EncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def make_model(self,encoder="residual_gru",decoder="concat"):
        model=TRNet(8,12,encoder_type=encoder,decoder_input=decoder)
        model.set_module(5)
        return model

    def test_optimized_replay_matches_original_outputs_and_gradients(self):
        reference=self.make_model()
        optimized=copy.deepcopy(reference)
        src=torch.tensor([1,2,3,1,4,2,1,3])
        dst=torch.tensor([2,3,4,4,2,5,2,5])
        t=torch.tensor([0,0,1,1,1,2,2,3])
        for start,end in ((0,4),(4,8)):
            legacy_update(reference,src[start:end],dst[start:end],t[start:end])
            optimized.update_mem(src[start:end],dst[start:end],t[start:end])
            for name in ("cumul_mem","prev_mem","last_t"):
                torch.testing.assert_close(getattr(reference.module,name),getattr(optimized.module,name))
        reference.module.cumul_mem.square().sum().backward()
        optimized.module.cumul_mem.square().sum().backward()
        for a,b in zip(reference.encoder.parameters(),optimized.encoder.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)

    def test_simultaneous_source_uses_snapshot_across_batches(self):
        model=self.make_model()
        initial=model.module.init_mem.clone()
        model.update_mem(torch.tensor([1]),torch.tensor([2]),torch.tensor([1]))
        expected=model.encoder(initial[2:3],initial[3:4],initial[2:3])
        model.update_mem(torch.tensor([2]),torch.tensor([3]),torch.tensor([1]))
        torch.testing.assert_close(model.module.cumul_mem[3:4],expected)
        # At a later time node 2 may propagate its newly accumulated memory.
        expected=model.encoder(model.module.cumul_mem[2:3],model.module.cumul_mem[4:5],initial[2:3])
        model.update_mem(torch.tensor([2]),torch.tensor([4]),torch.tensor([2]))
        torch.testing.assert_close(model.module.cumul_mem[4:5],expected)

    def test_all_variants_train_and_restore_state(self):
        for encoder in ("residual_gru","normalized_gru","diagonal","lru","gated_linear","mingru"):
            for decoder in ("concat","interaction","hybrid"):
                with self.subTest(encoder=encoder,decoder=decoder):
                    model=self.make_model(encoder,decoder)
                    original=model.module.snapshot_state()
                    model.update_mem(torch.tensor([1,2,3,2]),torch.tensor([2,3,4,5]),torch.tensor([1,1,2,3]))
                    pred=model(torch.tensor([1,2,4]),torch.tensor([4,5,1]))
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,torch.tensor([1.,1.,0.]))
                    loss.backward()
                    for name,param in model.named_parameters():
                        self.assertIsNotNone(param.grad,name)
                        self.assertTrue(torch.isfinite(param.grad).all(),name)
                    self.assertTrue(torch.isfinite(pred).all())
                    model.module.detach_module()
                    self.assertIsNone(model.module.cumul_mem.grad_fn)
                    model.module.restore_state(original)
                    for name,value in original.items():
                        torch.testing.assert_close(getattr(model.module,name),value)
                    model.update_mem(torch.tensor([1]),torch.tensor([2]),torch.tensor([1]))
                    model.reset_memory()
                    torch.testing.assert_close(model.module.cumul_mem,model.module.init_mem)
                    self.assertFalse(model.module.last_t.any())

    def test_decoder_shape_remains_fixed(self):
        for decoder in ("concat","interaction","hybrid"):
            model=self.make_model(decoder=decoder)
            self.assertEqual(model.decoder.mlp[0].in_features,24)
            self.assertEqual(model.decoder.mlp[0].out_features,12)
            self.assertEqual(model.decoder.mlp[2].out_features,1)


if __name__=="__main__":
    unittest.main()
