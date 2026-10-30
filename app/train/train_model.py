import argparse
from torch.utils.data import DataLoader
from model import TGAT
from model_train import ModelTrainer
from utils import DataUtils,GraphUtils,TrainUtils,TemporalGraphDataset,ModelUtils

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
    node_dim=n_node if one_hot else 32 # node_ft는 항상 None
    if edge_ft is not None:
        edge_dim=edge_ft.shape[1]
    else:
        edge_dim=32

    ### load TR_result
    batch_size=200
    train_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        purpose="train",
        batch_size=batch_size
    )
    val_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        purpose="val",
        batch_size=batch_size
    )

    ### set train data_loader and candidates, set val data_loader, candidates, TR_sample_loader
    n_sample=200
    n_pair=5
    train_df,val_df,_=TrainUtils.split_graph_df(graph_df=graph_df)
    train_dataset=TemporalGraphDataset(graph_df=train_df)
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
    train_candidates=train_dataset.dst.tolist()

    val_dataset=TemporalGraphDataset(graph_df=val_df)
    val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
    val_candidates=train_candidates+val_dataset.dst.tolist()
    val_sample_loader=TrainUtils.get_TR_sample_loader(
        graph_loader=val_loader,
        n_sample=n_sample,
        n_pair=n_pair,
        candidates=val_candidates,
        TR_label=val_TR_result["label"]
    )

    """
    2. Setting Model
    """
    ### 모델 관련 파라미터
    model_name=kwargs["model_name"]
    time_dim=32
    latent_dim=128
    embed_dim=32
    n_layer=2
    n_neighbor=10
    n_head=4

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

    ### set graph
    graph=GraphUtils.generate_model_graph(
        model_name=model_name,
        graph_df=graph_df,
        node_ft=node_ft,
        edge_ft=edge_ft,
        one_hot=one_hot,
        **graph_config
    )
    model.set_graph(graph=graph)

    """
    3. Train Model
    """
    ### 학습 관련 파라미터
    optimizer=f"adam"
    epoch=100
    early_stop=True
    patience=10
    batch_size=200
    lr=kwargs["lr"]
    model_config={
        "model_name":kwargs["model_name"],
        "optimizer":optimizer,
        "epoch":epoch,
        "early_stop":early_stop,
        "patience":patience,
        "batch_size":batch_size,
        "seed":seed,
        "lr":lr,
        "train_candidates":train_candidates,
        "n_sample":n_sample,
        "n_pair":n_pair
    }
    match model_name:
        case "TGAT":
            model=ModelTrainer.train(
                model=model,
                train_loader=train_loader,
                val_sample_loader=val_sample_loader,
                train_TR_label=train_TR_result["label"],
                **model_config
            )

    ### save model
    if kwargs["save_model"]:
        file_name=f"{model_name}_S{seed}_LR{lr}_B{batch_size}_{dataset_name}"
        ModelUtils.save_model_parameter(
            model=model,
            file_name=file_name,
            dataset_name=dataset_name
        )

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
            "ReaCH-TGN"
        ],
        default=f"TGAT"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0005,0.0001],default=0.0005)
    parser.add_argument("--one_hot",type=int,default=0)
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "CollegeMsg",
            "soc-sign-bitcoinalpha",
            "soc-sign-bitcoinotc"
        ],
        default="CollegeMsg"
    )
    parser.add_argument("--save_model",type=int,default=0)
    args=parser.parse_args()
    app_config={
        "model_name":args.model_name,
        "seed":args.seed,
        "lr":args.lr,
        "one_hot":args.one_hot,
        "dataset_name":args.dataset_name,
        "save_model":args.save_model
    }
    main(**app_config)