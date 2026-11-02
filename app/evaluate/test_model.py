import argparse
from torch.utils.data import DataLoader
from model import TGAT,TGN,CAW_N,GraphMixer,DyGFormer,TPNet,ReaCH_TGN,TRNet
from model_train import Model_Trainer,TPNet_Trainer,ReaCH_TGN_Trainer,TRNet_Trainer
from utils import DataUtils,GraphUtils,TrainUtils,TemporalGraphDataset,ModelUtils

"""
<< 공통 하이퍼 파라미터 >>
node_dim: 172
edge_dim: 172
time_dim: 100
latent_dim: 172
embed_dim: 

<< 공통 모델 관련 파라미터 >>
n_layer: 2
n_head: 2
n_neighbor: 10

<< 공통 모델 학습 관련 파라미터 >>
batch_size: 200
epoch (max_epoch): 100
learning_rate: 0.0001
drop_out: 0.1

<< 모델별 하이퍼 파라미터 >>
TGAT 
- neighbor_sampling: last

TGN
- mem_dim: 172
- msg_dim: 100
- msg_fn: mlp
- aggr_fn: last
- memory_updater: GRU

CAW-N
- n_walk: 64
- walk_len: 2
- alpha: 0.000001 (1e^-6)

GraphMixer 
- n_last_event: 50
- node_ft는 one-hot vec로 처리한다.

DyGFormer
- max_history_len: 64
- patch_size: 2
- co_dim: 50
- common_dim: 50

TPNet
- walk_len: 3
- time_decay_coef: 0.0000001 (1e^-7)
- n_last_event: 20
- mlp_mixer_n_layer: 2
- random_ft_dim: 128

ReaCH-TGN
- msg_fn: concat
- aggr_fn: last
- memory_updater: GRU
"""
def main(**kwargs):
    ### set seed
    seed=kwargs["seed"]
    TrainUtils.set_seed(seed=seed)

    """
    1. Setting Data
    """
    ### load data
    dataset_name=kwargs["dataset_name"]
    data=DataUtils.preprocess_SNAP_dataset(dataset_name=dataset_name)
    graph_df=data["graph_df"]
    n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
    node_ft=data["node_ft"]
    edge_ft=data["edge_ft"]
    one_hot=kwargs["one_hot"] 
    node_dim=n_node if one_hot else 172 # node_ft는 항상 None
    if edge_ft is not None:
        edge_dim=edge_ft.shape[1]
    else:
        edge_dim=172

    ### set train_loader and val_loader // load test_sample_loader
    batch_size=200
    train_df,val_df,test_df=TrainUtils.split_graph_df(graph_df=graph_df)
    train_dataset=TemporalGraphDataset(graph_df=train_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    val_dataset=TemporalGraphDataset(graph_df=val_df)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    test_dataset=TemporalGraphDataset(graph_df=test_df)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
    test_sample_loader=DataUtils.load_sample_loader(
        dataset_name=dataset_name,
        purpose="test",
        seed=seed,
        batch_size=batch_size
    )

    """
    2. Setting Model
    """
    ### 모델 관련 파라미터
    model_name=kwargs["model_name"]
    time_dim=100
    latent_dim=172
    embed_dim=172
    n_layer=2
    n_neighbor=10
    n_head=2

    ### set model and graph_config
    graph_config={}
    match model_name:
        case "TGAT":
            model=TGAT(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head
            )
        case "TGN":
            # TGN 하이퍼 파라미터
            msg_dim=100
            mem_dim=172
            msg_fn=f"mlp"
            aggr_fn=f"last"
            model=TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )
        case "CAW-N":
            # CAW-N 하이퍼 파라미터
            n_walk=64
            walk_len=2
            alpha=0.000001
            graph_config={
                "n_walk":n_walk,
                "walk_len":walk_len,
                "alpha":alpha
            }
            model=CAW_N(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                embed_dim=embed_dim,
                n_head=n_head,
                n_walk=n_walk,
                walk_len=walk_len,
                alpha=alpha
            )
        case "GraphMixer":
            # GraphMixer 하이퍼 파라미터
            n_last_event=50
            model=GraphMixer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                embed_dim=embed_dim,
                n_last_event=n_last_event
            )
        case "DyGFormer":
            # DyGFormer 하이퍼 파라미터
            co_dim=50
            common_dim=50
            max_history_len=64
            patch_size=2
            model=DyGFormer(
                node_dim=node_dim,
                edge_dim=edge_dim,
                latent_dim=latent_dim,
                time_dim=time_dim,
                co_dim=co_dim,
                common_dim=common_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_head=n_head,
                max_history_len=max_history_len,
                patch_size=patch_size
            )
        case "TPNet":
            # TPNet 하이퍼 파라미터
            random_ft_dim=128
            walk_len=3
            time_decay_coef=0.0000001
            n_last_event=20
            mlp_mixer_n_layer=2
            model=TPNet(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                random_ft_dim=random_ft_dim,
                embed_dim=embed_dim,
                walk_len=walk_len,
                time_decay_coef=time_decay_coef,
                n_last_event=n_last_event,
                mlp_mixer_n_layer=mlp_mixer_n_layer
            )
        case "ReaCH-TGN":
            # ReaCH-TGN 하이퍼 파라미터
            msg_dim=100
            mem_dim=172
            msg_fn=f"mlp"
            aggr_fn=f"last"
            model=ReaCH_TGN(
                node_dim=node_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                latent_dim=latent_dim,
                msg_dim=msg_dim,
                mem_dim=mem_dim,
                embed_dim=embed_dim,
                n_layer=n_layer,
                n_neighbor=n_neighbor,
                n_head=n_head,
                msg_fn=msg_fn,
                aggr_fn=aggr_fn
            )
        case "TRNet":
            pass

    ### set graph or module
    if model_name in ("TRNet"):
        model.set_module(n_node=n_node)
    else:
        graph=GraphUtils.generate_model_graph(
            model_name=model_name,
            graph_df=graph_df,
            node_ft=node_ft,
            edge_ft=edge_ft,
            node_dim=node_dim,
            edge_dim=edge_dim,
            one_hot=one_hot,
            **graph_config
        )
        model.set_graph(graph=graph)
        model.graph.set_in_adj_for_evaluate(
            train_loader=train_loader,
            val_loader=val_loader
        )

    """
    3. Load Model
    """
    batch_size=200
    lr=kwargs["lr"]
    model=ModelUtils.load_model_parameter(
        model=model,
        model_name=model_name,
        seed=seed,
        lr=lr,
        batch_size=batch_size,
        dataset_name=dataset_name
    )

    """
    4. Test Model
    """
    match model_name:
        case "TGAT"|"TGN"|"CAW-N"|"GraphMixer"|"DyGFormer":
            test_result=Model_Trainer.test(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader
            )
        case "TPNet":
            test_result=TPNet_Trainer.test(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader
            )
        case "ReaCH-TGN":
            test_result=ReaCH_TGN_Trainer.test(
                model=model,
                test_loader=test_loader,
                test_sample_loader=test_sample_loader
            )
        case "TRNet":
            pass

    print(f"Evaluation Result:")
    print(f"dataset: {dataset_name}")
    print(f"model: {model_name}")
    print(f"hypher parameter: seed={seed}, lr={lr}, batch_size={batch_size}")
    print(f"ACC: {test_result['acc']}")
    print(f"AP: {test_result['ap']}")
    print(f"AUC: {test_result['auc']}")

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=[
            "TGAT",
            "TGN",
            "CAW-N",
            "GraphMixer",
            "DyGFormer",
            "TPNet",
            "ReaCH-TGN",
            "TRNet"
        ],
        default=f"TGAT"
    )
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "CollegeMsg",
            "soc-sign-bitcoinalpha",
            "soc-sign-bitcoinotc"
        ],
        default="CollegeMsg"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0005,0.0001],default=0.0005)
    parser.add_argument("--one_hot",type=int,default=0)
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "dataset_name":args.dataset_name,
        "seed":args.seed,
        "lr":args.lr,
        "one_hot":args.one_hot
    }
    main(**app_config)