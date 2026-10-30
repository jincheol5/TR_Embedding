import argparse
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset

def main(**kwargs):
    ### set seed
    seed=kwargs["seed"]
    TrainUtils.set_seed(seed=seed)

    dataset_name=kwargs["dataset_name"]
    purpose=kwargs["purpose"]
    batch_size=kwargs["batch_size"]

    data=DataUtils.preprocess_SNAP_dataset(dataset_name=dataset_name)
    graph_df=data["graph_df"]
    TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        purpose=purpose,
        batch_size=batch_size
    )

    n_sample=200
    n_pair=5
    train_df,val_df,test_df=TrainUtils.split_graph_df(graph_df=graph_df)
    train_dataset=TemporalGraphDataset(graph_df=train_df)
    train_candidates=train_dataset.dst.tolist()

    val_dataset=TemporalGraphDataset(graph_df=val_df)
    val_candidates=train_candidates+val_dataset.dst.tolist()

    test_dataset=TemporalGraphDataset(graph_df=test_df)
    test_candidates=val_candidates+test_dataset.dst.tolist()

    if purpose=="val":
        val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
        sample_loader=TrainUtils.get_TR_sample_loader(
            graph_loader=val_loader,
            n_sample=n_sample,
            n_pair=n_pair,
            candidates=val_candidates,
            TR_label=TR_result["label"]
        )
    else: # test
        test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
        sample_loader=TrainUtils.get_TR_sample_loader(
            graph_loader=test_loader,
            n_sample=n_sample,
            n_pair=n_pair,
            candidates=test_candidates,
            TR_label=TR_result["label"]
        )

    DataUtils.save_sample_loader(
        sample_loader=sample_loader,
        dataset_name=dataset_name,
        purpose=purpose,
        batch_size=batch_size,
        seed=seed
    )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "CollegeMsg",
            "soc-sign-bitcoinalpha",
            "soc-sign-bitcoinotc"
        ],
        default="CollegeMsg"
    )
    parser.add_argument("--purpose",type=str,choices=["val","test"],default="test")
    parser.add_argument("--batch_size",type=int,default=200)
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name,
        "purpose":args.purpose,
        "batch_size":args.batch_size,
        "seed":args.seed
    }
    main(**app_config)