"""Algebra, temporal semantics, and lifecycle checks for matrix graph memories."""
import copy
import io
import math
import unittest

import torch

from model.TRNet import TRNet
from module.TRNet_matrix import MatrixLSTM,DeltaNet


class MatrixTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)

    def test_mlstm_stabilized_matches_unscaled_covariance(self):
        cell=MatrixLSTM(12,12,heads=3,key_dim=4).double()
        state=cell.initial_state(2,dtype=torch.float64)
        raw_C=torch.zeros_like(state["C"])
        raw_n=torch.zeros_like(state["n"])
        for _ in range(7):
            q,k=torch.randn(2,3,4,dtype=torch.float64),torch.randn(2,3,4,dtype=torch.float64)
            v=torch.randn(2,3,4,dtype=torch.float64)
            i_pre=torch.randn(2,3,dtype=torch.float64)
            f_pre=torch.randn(2,3,dtype=torch.float64)
            read,state=cell.recurrent_step(q,k,v,i_pre,f_pre,state)
            scaled_k=k/math.sqrt(4)
            f=torch.sigmoid(f_pre).unsqueeze(-1)
            i=i_pre.exp().unsqueeze(-1)
            raw_C=f.unsqueeze(-1)*raw_C+i.unsqueeze(-1)*scaled_k.unsqueeze(-1)*v.unsqueeze(-2)
            raw_n=f*raw_n+i*scaled_k
            torch.testing.assert_close(state["C"]*state["m"].exp().unsqueeze(-1),raw_C)
            torch.testing.assert_close(state["n"]*state["m"].exp(),raw_n)
            numerator=torch.einsum("bhk,bhkv->bhv",q,raw_C)
            denominator=torch.maximum(torch.einsum("bhk,bhk->bh",q,raw_n).abs().unsqueeze(-1),
                                      torch.ones(2,3,1,dtype=torch.float64))
            # NX-AI adds epsilon in stabilized coordinates.
            expected=numerator/(denominator+1e-6*state["m"].exp())
            torch.testing.assert_close(read,expected,rtol=1e-11,atol=1e-11)

    def test_mlstm_extreme_gates_have_finite_values_and_gradients(self):
        cell=MatrixLSTM(8,8,heads=2,key_dim=3)
        q=torch.randn(2,2,3,requires_grad=True)
        k=torch.randn(2,2,3,requires_grad=True)
        v=torch.randn(2,2,4,requires_grad=True)
        i=torch.tensor([[1000.,-1000.],[-1000.,1000.]],requires_grad=True)
        f=torch.tensor([[-1000.,-1000.],[1000.,1000.]],requires_grad=True)
        out,state=cell.recurrent_step(q,k,v,i,f,cell.initial_state(2))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))
        out.square().sum().backward()
        for value in (q,k,v,i,f):
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_delta_overwrites_key_without_forgetting_orthogonal_key(self):
        cell=DeltaNet(3,3,heads=1,key_dim=2).double()
        state=cell.initial_state(1,dtype=torch.float64)
        keys=[torch.tensor([[[1.,0.]]],dtype=torch.float64),
              torch.tensor([[[0.,1.]]],dtype=torch.float64)]
        values=[torch.tensor([[[2.,3.,4.]]],dtype=torch.float64),
                torch.tensor([[[5.,6.,7.]]],dtype=torch.float64),
                torch.tensor([[[-1.,-2.,-3.]]],dtype=torch.float64)]
        for key,value in ((keys[0],values[0]),(keys[1],values[1]),(keys[0],values[2])):
            prior=state["C"]
            output,state=cell.recurrent_step(key,key,value,torch.ones(1,1,dtype=torch.float64),state)
            direct=prior+key.unsqueeze(-1)*(value-torch.einsum("bhk,bhkv->bhv",key,prior)).unsqueeze(-2)
            torch.testing.assert_close(state["C"],direct)
            torch.testing.assert_close(output,value/math.sqrt(2))
        torch.testing.assert_close(state["C"][0,0,0],values[2][0,0])
        torch.testing.assert_close(state["C"][0,0,1],values[1][0,0])

    def test_same_time_source_does_not_expose_auxiliary_updates(self):
        for encoder in ("mlstm","deltanet"):
            with self.subTest(encoder=encoder):
                model=TRNet(8,10,encoder_type=encoder,event_backend="wave",matrix_heads=2,matrix_key_dim=3)
                model.set_module(4)
                control=copy.deepcopy(model)
                model.update_mem(torch.tensor([1]),torch.tensor([2]),torch.tensor([5]))
                # The tie crosses a batch boundary. Source 2 must still expose
                # its initial vector, although its matrix was already updated.
                for candidate in (model,control):
                    candidate.update_mem(torch.tensor([2]),torch.tensor([3]),torch.tensor([5]))
                torch.testing.assert_close(model.module.cumul_mem[3],control.module.cumul_mem[3])
                torch.testing.assert_close(model.module.aux_C[3],control.module.aux_C[3])
                for candidate in (model,control):
                    candidate.update_mem(torch.tensor([2]),torch.tensor([3]),torch.tensor([6]))
                self.assertGreater((model.module.aux_C[3]-control.module.aux_C[3]).abs().max().item(),1e-5)

    def test_wave_state_output_gradient_parity_and_lifecycle(self):
        events=torch.tensor([[1,2,0],[3,4,0],[2,3,0],[5,6,1],[1,3,1],[4,4,1],
                             [3,5,1],[6,2,2],[2,4,2],[4,1,3],[5,3,3],[2,6,3]])
        max_error=0.0
        for encoder in ("mlstm","deltanet"):
            for decoder in ("concat","interaction","hybrid","matrix_query","matrix_query_raw"):
                with self.subTest(encoder=encoder,decoder=decoder):
                    seq=TRNet(12,15,encoder_type=encoder,decoder_input=decoder,
                              matrix_heads=3,matrix_key_dim=4)
                    seq.set_module(6)
                    wave=copy.deepcopy(seq)
                    wave.event_backend="wave"
                    a_outputs,b_outputs=[],[]
                    for lo,hi in ((0,5),(5,8),(8,12)):
                        args=events[lo:hi].unbind(1)
                        seq.update_mem(*args)
                        wave.update_mem(*args)
                        for name,a in seq.module.snapshot_state().items():
                            b=getattr(wave.module,name)
                            max_error=max(max_error,(a-b).abs().max().item())
                            torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
                        s,d=torch.tensor([1,2,3,5]),torch.tensor([5,4,6,2])
                        a,b=seq(s,d),wave(s,d)
                        torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
                        a_outputs.append(a)
                        b_outputs.append(b)
                    torch.cat(a_outputs).square().mean().backward()
                    torch.cat(b_outputs).square().mean().backward()
                    for (name,a),(_,b) in zip(seq.named_parameters(),wave.named_parameters()):
                        self.assertIsNotNone(a.grad,name)
                        self.assertTrue(torch.isfinite(a.grad).all(),name)
                        torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-5,msg=name)
                    saved=wave.module.snapshot_state()
                    wave.module.detach_module()
                    for tensor in wave.module.auxiliary_state().values():
                        self.assertIsNone(tensor.grad_fn)
                    wave.reset_memory()
                    for tensor in wave.module.auxiliary_state().values():
                        self.assertEqual(torch.count_nonzero(tensor).item(),0)
                    wave.module.restore_state(saved)
                    for name,value in saved.items():
                        torch.testing.assert_close(getattr(wave.module,name),value)
                    stream=io.BytesIO()
                    torch.save(wave.state_dict(),stream)
                    stream.seek(0)
                    restored=TRNet(12,15,encoder_type=encoder,decoder_input=decoder,
                                   event_backend="wave",matrix_heads=3,matrix_key_dim=4)
                    restored.set_module(6)
                    restored.load_state_dict(torch.load(stream,weights_only=True))
                    torch.testing.assert_close(restored(s,d),wave(s,d))
                    self.assertEqual(restored.module.aux_C.shape,(7,3,4,4))
        print(f"Matrix wave maximum state absolute error={max_error:.3g}")

    def test_query_read_addresses_destination_memory_and_differentiates(self):
        for encoder in ("mlstm","deltanet"):
            with self.subTest(encoder=encoder):
                model=TRNet(12,15,encoder_type=encoder,decoder_input="matrix_query",
                            event_backend="wave",matrix_heads=3,matrix_key_dim=4)
                model.set_module(4)
                with torch.no_grad():
                    model.module.aux_C.normal_()
                model.module.aux_C.requires_grad_()
                src,dst=torch.tensor([1,1]),torch.tensor([2,3])
                prior=model(src,dst)
                self.assertEqual(model.decoder.mlp[0].in_features,36)
                prior.square().sum().backward()
                self.assertGreater(model.encoder.q_proj.weight.grad.abs().sum().item(),0)
                self.assertGreater(model.module.aux_C.grad[2].abs().sum().item(),0)
                self.assertEqual(model.module.aux_C.grad[4].abs().sum().item(),0)
                with torch.no_grad():
                    model.module.aux_C[2].add_(torch.randn_like(model.module.aux_C[2]))
                    after=model(src,dst)
                self.assertGreater((after[0]-prior[0]).abs().item(),1e-6)
                torch.testing.assert_close(after[1],prior[1],rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,"matrix-memory"):
            TRNet(12,15,encoder_type="mingru",decoder_input="matrix_query")

    def test_raw_query_read_matches_retrieval_before_output_transform(self):
        for encoder in ("mlstm","deltanet"):
            with self.subTest(encoder=encoder):
                model=TRNet(12,15,encoder_type=encoder,decoder_input="matrix_query_raw",
                            event_backend="wave",matrix_heads=3,matrix_key_dim=4)
                model.set_module(4)
                state=model.encoder.initial_state(2)
                state["C"]=torch.randn_like(state["C"],requires_grad=True)
                inputs=model.module.init_mem[torch.tensor([1,2])]
                normalized=model.encoder.input_norm(inputs)
                q=model.encoder.q_proj(normalized).reshape(2,3,4)
                if encoder=="mlstm":
                    expected=model.encoder.retrieve(q,state).reshape(2,12)
                else:
                    q=torch.nn.functional.normalize(torch.nn.functional.silu(q),dim=-1,eps=1e-6)
                    expected=(torch.einsum("bhk,bhkv->bhv",q,state["C"])/2).reshape(2,12)
                result=model.encoder.read_state(inputs,inputs,state,raw=True)
                torch.testing.assert_close(result,expected,rtol=0,atol=0)
                result.square().sum().backward()
                self.assertGreater(state["C"].grad.abs().sum().item(),0)
                self.assertGreater(model.encoder.q_proj.weight.grad.abs().sum().item(),0)
                self.assertIsNone(model.encoder.output_proj.weight.grad)
                self.assertEqual(model.decoder.mlp[0].in_features,36)


if __name__=="__main__":
    unittest.main()
