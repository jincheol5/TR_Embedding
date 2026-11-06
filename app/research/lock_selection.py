"""Lock a validation-only model decision before accessing held-out scores."""
import argparse
import json
from pathlib import Path
import time

from .run import OUT,ROOT,atomic_json,config_id
from .summarize import aggregate,load_records,select_winners,render_report,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages",nargs="+",default=["confirmation","confirmation_linear"])
    parser.add_argument("--minimum-seeds",type=int,default=3)
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--prefix",default="",help="Preserve previous artifacts in a separate study namespace")
    args=parser.parse_args()
    prefix=(args.prefix+"_") if args.prefix else ""
    args.selection=args.selection or OUT/f"{prefix}selection.json"
    if not prefix and args.selection!=OUT/"selection.json":
        raise ValueError("Use --prefix with a custom selection to preserve original artifacts")
    if args.selection.exists():
        raise FileExistsError("Selection is already locked; do not change it after test evaluation")
    records=[]
    for stage in args.stages:
        stage_records=load_records(OUT/"results"/stage)
        manifest=json.loads((OUT/"results"/stage/"manifest.json").read_text())
        expected={config_id(config) for config in manifest["configs"]}
        completed={config_id(record["config"]) for record in stage_records}
        if expected!=completed:
            raise ValueError(f"Not all configurations are complete in stage {stage}")
        records.extend(stage_records)
    if not records or any("test" in record for record in records):
        raise ValueError("Use completed validation-only records")
    # Stages differ only to permit independent job queues. Never merge budgets.
    budgets={(record["requested_epochs"],record["patience"],record["grad_clip"])
             for record in records}
    if len(budgets)!=1:
        raise ValueError("Confirmation stages must use the same training budget")
    source_paths=[str(Path(record["path"]).resolve().relative_to(ROOT)) for record in records]
    for record in records:
        record["source_stage"]=record["stage"]
        record["stage"]="confirmation_comparison"
    summaries=aggregate(records)
    if any(summary["n_runs"]<args.minimum_seeds for summary in summaries):
        raise ValueError("Some candidate configurations do not have all required seeds")
    winners=select_winners(summaries,args.minimum_seeds)
    winner=winners["confirmation_comparison"]
    selected=winner["config"]
    selected_runs=[record for record in records if
                   {key:value for key,value in record["config"].items() if key!="seed"}==selected]
    best_run=max(selected_runs,key=lambda record:tuple(record["validation"][name]
                                                       for name in ("auc","ap","acc")))
    payload={"selected_config":selected,
             "selection_rule":"3-seed mean validation AUC, then AP, then ACC; all comparisons use the same budget",
             "selected_validation":winner["validation"],"source_stages":args.stages,
             "result_paths":source_paths,"created_unix":time.time(),
             "best_single_seed_by_validation":best_run["config"]["seed"],
             "best_checkpoint_by_validation":best_run["checkpoint"],
             "test_used_for_selection":False,
             "candidates":[{"config":item["config"],"validation":item["validation"],
                            "seeds":item["seeds"]} for item in summaries]}
    atomic_json(args.selection,payload)
    atomic_json(OUT/f"{prefix}best_config.json",payload)
    atomic_json(OUT/f"{prefix}best_config_runs.json",[record["config"] for record in selected_runs])
    report=OUT/f"{prefix}CONFIRMATION.md"
    report.write_text(render_report(summaries,winners,args.minimum_seeds))
    write_csv(summaries,report.with_suffix(".csv"))
    atomic_json(report.with_suffix(".json"),{"summaries":summaries,"winners":winners})
    print(json.dumps({"selection":str(args.selection),"selected_config":selected,
                      "validation":winner["validation"]},ensure_ascii=False))


if __name__=="__main__":
    main()
