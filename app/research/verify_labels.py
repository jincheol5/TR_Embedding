"""Independently audit cached strict-time TR labels with integer bitsets.

Run from the repository root. This reads data/caches without changing them.
"""

import argparse
import json
from pathlib import Path

import torch

from utils import DataUtils, TrainUtils
from utils.data_utils import BASE_PATH


def audit(dataset="CollegeMsg", seeds=(1, 2, 3), batch_size=200):
    frame = DataUtils.preprocess_SNAP_dataset(dataset)["graph_df"]
    partitions = TrainUtils.split_graph_df(frame)
    n_node = int(max(frame.u.max(), frame.i.max()))
    ancestors = [1 << node for node in range(n_node + 1)]
    previous = {}
    last_t = {}
    results = []
    for purpose, partition in zip(("train", "val", "test"), partitions):
        samples = {
            seed: DataUtils.load_sample_loader(dataset, purpose, seed, batch_size)
            for seed in seeds
        } if purpose != "train" else {}
        cache_path = (
            Path(BASE_PATH) / "temporal_graph" / "SNAP" / dataset / "TR_result"
            / f"{dataset}_{purpose}_B{batch_size}.pt"
        )
        label = torch.load(cache_path, mmap=True, weights_only=True)["label"] if purpose == "train" else None
        checked = 0
        mismatches = 0
        for batch_idx, start in enumerate(range(0, len(partition), batch_size)):
            batch = partition.iloc[start:start + batch_size]
            for src, dst, timestamp in batch[["u", "i", "t"]].itertuples(index=False, name=None):
                src_state = previous[src] if last_t.get(src) == timestamp else ancestors[src]
                if last_t.get(dst) != timestamp:
                    previous[dst] = ancestors[dst]
                ancestors[dst] |= src_state
                last_t[dst] = timestamp
            if samples:
                queries = [
                    zip(sample[batch_idx]["src"].tolist(),
                        sample[batch_idx]["dst"].tolist(),
                        sample[batch_idx]["label"].tolist())
                    for sample in samples.values()
                ]
            else:
                # Independently chosen entries, including recently updated nodes.
                sources = [1, min(10, n_node), min(100, n_node), n_node // 2, n_node]
                destinations = [int(batch.i.iloc[0]), int(batch.i.iloc[-1]), min(200, n_node), max(1, n_node // 3), max(1, n_node - 1)]
                queries = [zip(sources, destinations, label[batch_idx, sources, destinations].tolist())]
            for query in queries:
                for src, dst, truth in query:
                    checked += 1
                    if bool((ancestors[dst] >> src) & 1) != bool(truth):
                        mismatches += 1
        results.append({"partition": purpose, "checked": checked, "mismatches": mismatches})
    report = {
        "dataset": dataset,
        "nodes": n_node,
        "events": len(frame),
        "partition_events": [len(partition) for partition in partitions],
        "seeds": list(seeds),
        "timestamp_ties_at_split": [
            bool(left.t.iloc[-1] == right.t.iloc[0])
            for left, right in zip(partitions, partitions[1:])
        ],
        "results": results,
    }
    if any(result["mismatches"] for result in results):
        raise AssertionError(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CollegeMsg")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.dataset, args.seeds)
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
