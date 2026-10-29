import argparse
from torch.utils.data import DataLoader
from graph import TemporalGraph
from utils import DataUtils,GraphUtils,TrainUtils,TemporalGraphDataset

def main(**kwargs):
    dataset_name=kwargs["dataset_name"]
    batch_size=kwargs["batch_size"]

    ### load data
    graph_data=DataUtils.preprocess_SNAP_dataset(dataset_name=dataset_name)
    graph_df=graph_data["graph_df"]
    graph=TemporalGraph(graph_df=graph_df)
    n_node=graph.n_node

    ### split train (70%), val (15%), test (15%)
    train_df,val_df,test_df=TrainUtils.split_graph_df(graph_df=graph_df)

    ### compute and save TR_result
    gamma_table=None
    for purpose,graph_df in zip(["train","val","test"],[train_df,val_df,test_df]):
        graph_dataset=TemporalGraphDataset(graph_df=graph_df)
        graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)
        result=GraphUtils.get_TR_result_using_Time_Centric(
            n_node=n_node,
            graph_loader=graph_loader,
            gamma_table=gamma_table,
        )
        TR_result=result["TR_result"]
        gamma_table=result["gamma_table"]
        DataUtils.save_TR_result_to_pt(
            TR_result=TR_result,
            dataset_name=dataset_name,
            purpose=purpose,
            batch_size=batch_size
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
    parser.add_argument("--batch_size",type=int,default=200)
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name,
        "batch_size":args.batch_size
    }
    main(**app_config)
