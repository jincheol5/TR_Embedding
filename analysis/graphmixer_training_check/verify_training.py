"""Exercise actual app with full data for one epoch; do not save model externally."""
import os
os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import time
import torch
from app.train import train_model as app

OUT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    original_train=app.Model_Trainer.train
    original_validate=app.Model_Trainer.validate
    original_step=torch.optim.Adam.step
    result={'train_forward_batches':0,'validation_forward_batches':0,'optimizer_steps':0,
            'device':'cpu','epochs':1,'one_hot':1,'seed':1,'lr':0.0001,
            'torch':torch.__version__}
    def forward_check(model,args,output):
        assert torch.isfinite(output).all(), 'Nonfinite logits'
        result['train_forward_batches' if model.training else 'validation_forward_batches']+=1
    def step_check(optimizer,*args,**kwargs):
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), 'Nonfinite gradient'
        output=original_step(optimizer,*args,**kwargs)
        result['optimizer_steps']+=1
        return output
    def validate(*args,**kwargs):
        metrics=original_validate(*args,**kwargs)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        result['validation_metrics']=metrics
        return metrics
    def train(*args,**kwargs):
        model=kwargs['model']
        assert model.edge_dim==model.graph.edge_dim
        assert model.node_dim==model.graph.node_dim
        result['dimensions']={'model_node':model.node_dim,'graph_node':model.graph.node_dim,
                              'model_edge':model.edge_dim,'graph_edge':model.graph.edge_dim,
                              'time':model.time_dim,'embedding':model.embed_dim,'history':model.n_last_event}
        kwargs['epoch']=1
        model.register_forward_hook(forward_check)
        return original_train(*args,**kwargs)
    app.Model_Trainer.train=staticmethod(train)
    app.Model_Trainer.validate=staticmethod(validate)
    torch.optim.Adam.step=step_check
    start=time.monotonic()
    app.main(model_name='GraphMixer',seed=1,lr=0.0001,one_hot=1,dataset_name='CollegeMsg',save_model=0)
    assert result['train_forward_batches']==210
    assert result['validation_forward_batches']==45
    assert result['optimizer_steps']==210
    result['seconds']=round(time.monotonic()-start,2)
    result['status']='PASS'
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
