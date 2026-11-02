"""Equal cumulative-memory-budget bucket sketch experiment (no model changes)."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/trnet_matplotlib')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score
from utils import DataUtils,TrainUtils
from module.TRNet_module import TRNet_Module

OUT=Path(__file__).resolve().parent

class BucketMemory:
    def __init__(self,initial,buckets,assignment):
        self.initial=initial.astype(np.float32)
        n,d=initial.shape
        self.assignment=assignment
        self.q=np.zeros((n,buckets,d),dtype=np.float32)
        self.s=np.full((n,buckets),-np.inf,dtype=np.float32)
        norms=np.linalg.norm(self.initial,axis=1)
        ids=np.flatnonzero(norms>0)
        self.q[ids,assignment[ids]]=self.initial[ids]/norms[ids,None]
        self.s[ids,assignment[ids]]=np.log(norms[ids])
        self.old_q=np.zeros_like(self.q); self.old_s=np.full_like(self.s,-np.inf)
        self.stamp=np.full(n,-1,dtype=np.int64)

    def update(self,events):
        for u,v,t in events:
            if self.stamp[v]!=t:
                self.old_q[v]=self.q[v]; self.old_s[v]=self.s[v]; self.stamp[v]=t
            sq=self.old_q[u] if self.stamp[u]==t else self.q[u]
            ss=self.old_s[u] if self.stamp[u]==t else self.s[u]
            common=np.maximum(self.s[v],ss)
            active=np.isfinite(common)
            if not active.any():
                continue
            z=(np.exp(self.s[v,active]-common[active])[:,None]*self.q[v,active]
               +np.exp(ss[active]-common[active])[:,None]*sq[active])
            norm=np.linalg.norm(z,axis=1)
            valid=norm>0
            z[valid]/=norm[valid,None]
            scales=np.full(norm.shape,-np.inf,dtype=np.float32)
            scales[valid]=common[active][valid]+np.log(norm[valid])
            self.q[v,active]=z; self.s[v,active]=scales

    def scores(self,sample):
        src=sample['src'].numpy(); dst=sample['dst'].numpy()
        bucket=self.assignment[src]
        r=self.initial[src]; q=self.q[dst,bucket]; scale=self.s[dst,bucket]
        dot=(r*q).sum(-1)
        cosine=dot/np.linalg.norm(r,axis=1)
        with np.errstate(divide='ignore',invalid='raise'):
            full=np.sign(dot)*np.logaddexp(0,np.log(np.abs(dot))+scale)
        assert np.isfinite(dot).all() and np.isfinite(cosine).all() and np.isfinite(full).all()
        return np.stack([sample['label'].numpy(),dot,cosine,full],axis=1)

def verify():
    # Independent direct-sum reference, including same-time events across calls.
    initial=np.eye(5,dtype=np.float32)
    assignment=np.array([0,1,0,1,0])
    x=BucketMemory(initial,2,assignment)
    reference=np.zeros((5,2,5),dtype=np.float32)
    reference[np.arange(5),assignment]=initial
    events=np.array([(1,2,1),(2,3,1),(1,3,1),(3,4,2),(4,2,3),(2,2,4)])
    for t in np.unique(events[:,2]):
        old=reference.copy()
        for u,v,_ in events[events[:,2]==t]: reference[v]+=old[u]
    for event in events: x.update([event])
    np.testing.assert_allclose(x.q*np.exp(x.s)[:,:,None],reference,rtol=2e-6,atol=2e-6)
    # Compare single-bucket engine against repository implementation.
    torch.manual_seed(1); model=TRNet_Module(4,5)
    x=BucketMemory(model.init_mem_vec.numpy().copy(),1,np.zeros(5,dtype=int))
    for event in events:
        x.update([event]); u,v,t=event
        model.update_module_vec(torch.tensor([u]),torch.tensor([v]),torch.tensor([t]))
    np.testing.assert_allclose(x.q[:,0],model.cumul_mem_vec.numpy(),rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(x.s[:,0],model.cumul_log_scale.numpy(),rtol=2e-5,atol=2e-6)
    print('PASS: direct sum / timestamp snapshots / repository single-bucket equivalence',flush=True)

def metrics(y,s):
    return {'auc':float(roc_auc_score(y,s)),'ap':float(average_precision_score(y,s)),
            'positive_mean':float(s[y==1].mean()),'negative_mean':float(s[y==0].mean())}

def main():
    verify(); torch.set_num_threads(1)
    frame=DataUtils.preprocess_SNAP_dataset('CollegeMsg')['graph_df']
    train,val,_=TrainUtils.split_graph_df(frame)
    samples=DataUtils.load_sample_loader('CollegeMsg','val',200)
    n=int(max(frame.u.max(),frame.i.max()))+1
    train_events=train[['u','i','t']].to_numpy()
    val_events=val[['u','i','t']].to_numpy()
    # Budget includes BOTH direction coordinates and log-scale scalars.
    # Snapshot has the same budget again for every configuration.
    configs=[(33,1)]+[(budget,k) for budget in [64,256,1024] for k in [1,4,16]]
    reports=[]; saved={}
    for budget,k in configs:
        dim=budget//k-1
        for seed in [1,2,3]:
            torch.manual_seed(seed)
            initial=torch.randn(n,dim).numpy()/np.sqrt(np.float32(dim))
            rng=np.random.default_rng(seed+10000)
            assignment=rng.integers(0,k,size=n)
            x=BucketMemory(initial,k,assignment)
            x.update(train_events)
            chunks=[]
            for i,sample in enumerate(samples):
                batch=val_events[i*200:(i+1)*200]
                assert (sample['query_t'].numpy()==batch[-1,2]).all()
                x.update(batch); chunks.append(x.scores(sample))
            data=np.concatenate(chunks); y=data[:,0]
            report={'budget_floats':budget,'buckets':k,'dim_per_bucket':dim,'seed':seed,
                    'scores':{name:metrics(y,data[:,col]) for col,name in enumerate(['dot','cosine','full_dot'],1)}}
            reports.append(report); saved[f'budget{budget}_b{k}_seed{seed}']=data
            print(json.dumps({'budget':budget,'buckets':k,'dim':dim,'seed':seed,
                'auc':{a:round(b['auc'],5) for a,b in report['scores'].items()}}),flush=True)
        (OUT/'summary.json').write_text(json.dumps({'dataset':'CollegeMsg','samples':9000,
            'train_events':len(train),'val_events':len(val),'dtype':'float32',
            'budget_definition':'per-node cumulative state: K*(d+1) float32 values; same-size snapshot; excludes immutable initial vectors and integer hashes',
            'protocol':'Strict time increasing propagation, snapshots persist across batches; saved validation pairs; no decoder training. Independent seeded hash per run; query uses source-assigned destination bucket.',
            'results':reports},indent=2)+'\n')
    np.savez_compressed(OUT/'scores.npz',**saved,columns=np.array(['label','dot','cosine','full_dot']))
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,name in zip(axes,['dot','cosine','full_dot']):
        for budget in [64,256,1024]:
            means=[]; deviations=[]
            for k in [1,4,16]:
                v=[r['scores'][name]['auc'] for r in reports if r['budget_floats']==budget and r['buckets']==k]
                means.append(np.mean(v)); deviations.append(np.std(v))
            ax.errorbar([1,4,16],means,yerr=deviations,marker='o',capsize=4,label=f'{budget} floats / node')
        ax.axhline(.5,color='gray',linestyle='--'); ax.set_xscale('log',base=4)
        ax.set_xticks([1,4,16],['1','4','16']); ax.set_xlabel('Buckets'); ax.set_title(name); ax.legend(fontsize=8)
    axes[0].set_ylabel('ROC AUC (mean +/- seed std)'); fig.tight_layout(); fig.savefig(OUT/'auc_comparison.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,k in zip(axes,[1,4,16]):
        data=saved[f'budget256_b{k}_seed1']; bins=np.linspace(-1,1,61)
        for label,color,title in [(0,'tab:orange','Unreachable'),(1,'tab:blue','Reachable')]:
            ax.hist(data[data[:,0]==label,2],bins=bins,density=True,histtype='step',label=title,color=color)
        ax.set_title(f'256 floats: {k} buckets x {256//k-1} dims\ncosine AUC={roc_auc_score(data[:,0],data[:,2]):.4f}')
        ax.set_xlabel('Cosine'); ax.legend(fontsize=8)
    axes[0].set_ylabel('Density'); fig.tight_layout(); fig.savefig(OUT/'distributions.png',dpi=160)
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
