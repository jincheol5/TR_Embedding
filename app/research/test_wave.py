"""Wave scheduling preserves sequential temporal states and gradients.

Run: python -m unittest app.research.test_wave -v
"""
import copy
import unittest

import torch

from model.TRNet import TRNet


ENCODERS=("residual_gru","normalized_gru","diagonal","lru","gated_linear","mingru")
DECODERS=("concat","interaction","hybrid")


class WaveTests(unittest.TestCase):
    def test_states_outputs_gradients_across_batches(self):
        torch.set_num_threads(1)
        # Ties straddle boundaries; shared sources, destinations, self-loops,
        # disconnected edges, fan-out, fan-in, and later propagation are covered.
        events=torch.tensor([
            [1,2,0],[3,4,0],[5,6,0],[2,3,0],[7,8,0],
            [1,3,1],[1,4,1],[5,4,1],[4,4,1],[4,8,1],
            [8,6,1],[2,7,1],[3,5,2],[6,7,2],[7,8,2],
            [8,1,3],[3,2,3],[5,6,3],[2,4,3],[6,8,3],
        ])
        max_state_error=0.0
        max_output_error=0.0
        max_grad_error=0.0
        for encoder in ENCODERS:
            for decoder in DECODERS:
                with self.subTest(encoder=encoder,decoder=decoder):
                    torch.manual_seed(919)
                    seq=TRNet(16,20,encoder_type=encoder,decoder_input=decoder)
                    seq.set_module(8)
                    wave=copy.deepcopy(seq)
                    wave.event_backend="wave"
                    seq_outputs=[]
                    wave_outputs=[]
                    query_s=torch.tensor([1,2,3,5,7,8])
                    query_d=torch.tensor([4,5,6,2,1,3])
                    for start,end in ((0,7),(7,11),(11,20)):
                        s,d,t=events[start:end].unbind(1)
                        seq.update_mem(s,d,t)
                        wave.update_mem(s,d,t)
                        for name in ("cumul_mem","prev_mem","last_t"):
                            a,b=getattr(seq.module,name),getattr(wave.module,name)
                            max_state_error=max(max_state_error,(a-b).abs().max().item())
                            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
                        a,b=seq(query_s,query_d),wave(query_s,query_d)
                        max_output_error=max(max_output_error,(a-b).abs().max().item())
                        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
                        seq_outputs.append(a)
                        wave_outputs.append(b)
                    torch.cat(seq_outputs).square().mean().backward()
                    torch.cat(wave_outputs).square().mean().backward()
                    for (name,a),(_,b) in zip(seq.named_parameters(),wave.named_parameters()):
                        self.assertIsNotNone(a.grad,name)
                        self.assertIsNotNone(b.grad,name)
                        max_grad_error=max(max_grad_error,(a.grad-b.grad).abs().max().item())
                        torch.testing.assert_close(a.grad,b.grad,rtol=3e-5,atol=3e-6,msg=name)
        print(f"Wave parity max absolute errors: state={max_state_error:.3g}, output={max_output_error:.3g}, gradient={max_grad_error:.3g}")

    def test_empty_batch_is_noop(self):
        model=TRNet(8,12,event_backend="wave")
        model.set_module(5)
        before=model.module.snapshot_state()
        empty=torch.empty(0,dtype=torch.long)
        model.update_mem(empty,empty,empty)
        for name,value in before.items():
            torch.testing.assert_close(getattr(model.module,name),value)


if __name__=="__main__":
    unittest.main()
