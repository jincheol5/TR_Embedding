"""Replay current TRNet memory on CollegeMsg validation; no decoder training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from module.TRNet_module import TRNet_Module
from utils import DataUtils, TrainUtils

OUT=Path(__file__).resolve().parent

def stats(y,score):
    return {'auc':float(roc_auc_score(y,score)), 'ap':float(average_precision_score(y,score)),
            'positive':describe(score[y==1]),'negative':describe(score[y==0])}

def describe(v):
    return {'mean':float(v.mean()),'std':float(v.std()),'q05':float(np.quantile(v,.05)),
            'median':float(np.median(v)),'q95':float(np.quantile(v,.95))}

def main():
    torch.set_num_threads(1)
    frame=DataUtils.preprocess_SNAP_dataset('CollegeMsg')['graph_df']
    train,val,_=TrainUtils.split_graph_df(frame)
    samples=DataUtils.load_sample_loader('CollegeMsg','val',200)
    assert len(samples)==(len(val)+199)//200
    results=[]; arrays=[]
    for seed in [1,2,3]:
        torch.manual_seed(seed)
        module=TRNet_Module(int(max(frame.u.max(),frame.i.max())),32)
        for start in range(0,len(train),200):
            batch=train.iloc[start:start+200]
            module.update_module_vec(*(torch.tensor(batch[c].to_numpy()) for c in ['u','i','t']))
        rows=[]
        for idx,sample in enumerate(samples):
            batch=val.iloc[idx*200:(idx+1)*200]
            assert (sample['query_t']==int(batch.t.max())).all()
            module.update_module_vec(*(torch.tensor(batch[c].to_numpy()) for c in ['u','i','t']))
            src=sample['src']; dst=sample['dst']
            r=module.get_init_mem_vec(src); q=module.get_cumul_mem_vec(dst)
            dot=(r*q).sum(-1)
            cosine=dot/torch.linalg.vector_norm(r,dim=-1)
            scale=module.get_cumul_log_scale(dst)
            # sign(x)*log(1+abs(x)), where x=<initial_src, actual_cumulative_dst>.
            # Strictly increasing transform: preserves ranking/AUC without exp(scale).
            logabs=dot.abs().log()+scale
            full=dot.sign()*torch.logaddexp(torch.zeros_like(logabs),logabs)
            values=torch.stack([sample['label'],dot,cosine,full,scale,
                                torch.full_like(dot,idx)],dim=-1).numpy()
            assert np.isfinite(values).all()
            rows.append(values)
        data=np.concatenate(rows); arrays.append(data)
        y=data[:,0]
        report={'seed':seed,'n':len(y),'positive_n':int(y.sum()),'scores':{},'quarters':[]}
        for col,name in enumerate(['direction_dot','cosine','signed_log_full_dot','dst_log_scale'],1):
            report['scores'][name]=stats(y,data[:,col])
        for batch_ids in np.array_split(np.arange(len(samples)),4):
            part=data[np.isin(data[:,5],batch_ids)]
            report['quarters'].append({'batch_start':int(batch_ids[0]),'batch_end':int(batch_ids[-1]),
                'cosine_auc':float(roc_auc_score(part[:,0],part[:,2])),
                'full_dot_auc':float(roc_auc_score(part[:,0],part[:,3]))})
        results.append(report)
        print(json.dumps({'seed':seed,'auc':{k:v['auc'] for k,v in report['scores'].items()}}),flush=True)
    np.savez_compressed(OUT/'scores.npz',**{f'seed_{s}':a for s,a in zip([1,2,3],arrays)},
                        columns=np.array(['label','direction_dot','cosine','signed_log_full_dot','dst_log_scale','batch_idx']))
    payload={'dataset':'CollegeMsg','mem_dim':32,'batch_size':200,'train_events':len(train),
             'val_events':len(val),'protocol':'Fresh random memory per seed, replay train then val; stored validation samples; CPU float32; no trained checkpoint. Seeds do not reproduce training RNG consumption.',
             'results':results}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for row,(seed,data) in enumerate(zip([1,2,3],arrays)):
        for col,(score_col,title) in enumerate([(2,'Cosine'),(3,'Signed log full dot'),(4,'Destination log norm')]):
            ax=axes[row,col]; bins=np.linspace(data[:,score_col].min(),data[:,score_col].max(),61)
            for label,color,name in [(0,'tab:orange','Unreachable'),(1,'tab:blue','Reachable')]:
                ax.hist(data[data[:,0]==label,score_col],bins=bins,density=True,histtype='step',linewidth=1.5,color=color,label=name)
            auc=roc_auc_score(data[:,0],data[:,score_col]); ax.set_title(f'Seed {seed}: {title}\nAUC = {auc:.4f}')
            ax.set_ylabel('Density'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/'distributions.png',dpi=160); plt.close(fig)

if __name__=='__main__':
    main()
