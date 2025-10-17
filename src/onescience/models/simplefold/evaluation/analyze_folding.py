#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import json
import argparse
import concurrent.futures
import subprocess
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd


# to download the docker image, refer to: 
# https://git.scicore.unibas.ch/schwede/openstructure#docker

OST_COMPARE_STRUCTURE = r"""
IMAGE_NAME=registry.scicore.unibas.ch/schwede/openstructure:2.9.1

command="compare-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--min-pep-length 4 \
--min-nuc-length 4 \
-o {output_path} \
--lddt --bb-lddt \
--ics --ips --rigid-scores --patch-scores --tm-score"

docker run --rm -v $(pwd):/home $IMAGE_NAME $command
"""


METRICS = ["lddt", "bb_lddt", "tm_score", "rmsd", "oligo_gdtts"]


def evaluate_structure(
    name: str,
    pred: Path,
    reference: Path,
    outdir: str,
) -> None:
    """Evaluate the structure."""
    # Evaluate polymer metrics
    out_path = Path(outdir) / f"{name}.json"

    if out_path.exists():
        print(
            f"Skipping recomputation of {name} as protein json file already exists"
        )
    else:
        subprocess.run(
            OST_COMPARE_STRUCTURE.format(
                model_file=str(pred),
                reference_file=str(reference),
                output_path=str(out_path),
            ),
            shell=True,
            check=False,
            capture_output=True,
        )


def run_eval(args):
    # Aggregate the predictions and references
    files = list(args.data.iterdir())
    names = {f.stem: f for f in files}

    # Create the output directory
    args.outdir.mkdir(parents=True, exist_ok=True)

    first_item = True
    with concurrent.futures.ThreadPoolExecutor(args.max_workers) as executor:
        futures = []
        for name, folder in names.items():
            pred_path = args.sample / f"{name}_sampled_0.cif"
            ref_path = args.data / f"{name}.cif"
            print(f"Processing {ref_path} and {pred_path}")

            if first_item:
                # Evaluate the first item in the first prediction
                # Ensures that the docker image is downloaded
                evaluate_structure(
                    name=f"{name}",
                    pred=str(pred_path),
                    reference=str(ref_path),
                    outdir=str(args.outdir),
                )
                first_item = False
            else:
                future = executor.submit(
                    evaluate_structure,
                    name=f"{name}",
                    pred=str(pred_path),
                    reference=str(ref_path),
                    outdir=str(args.outdir),
                )
                futures.append(future)

        # Wait for all tasks to complete
        with tqdm(total=len(futures)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def compute_metrics(preds, evals, name):
    metrics = {}

    # Load eval file
    eval_file = Path(evals) / f"{name}.json"
    with eval_file.open("r") as f:
        eval_data = json.load(f)
        for metric_name in METRICS:
            if metric_name in eval_data:
                metrics.setdefault(metric_name, []).append(eval_data[metric_name])

    # Get results
    oracle = {k: min(v) if k == "rmsd" else max(v) for k, v in metrics.items()}

    results = {}
    for metric_name in metrics:
        if metric_name.startswith("len_"):
            continue
        else:
            l = 1
        results[metric_name] = {
            "res": oracle[metric_name],
            "len": l
        }

    return results


def bootstrap_ci(series, n_boot=1000, alpha=0.05):
    """
    Compute 95% bootstrap confidence intervals for the mean of 'series'.
    """
    n = len(series)
    boot_means = []
    boot_medians = []
    # Perform bootstrap resampling
    for _ in range(n_boot):
        sample = series.sample(n, replace=True)
        boot_means.append(sample.mean())
        boot_medians.append(sample.median())

    boot_means = np.array(boot_means)
    mean_val = np.mean(boot_means)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return mean_val, lower, upper, np.median(boot_medians)


def aggregate_eval(args):
    simplefold_datas = args.data
    simplefold_evals = args.outdir
    # Load preds and make sure we have predictions for all models
    simplefold_datas_names = {x.name.split('.')[0]: x for x in Path(simplefold_datas).iterdir() if not x.name.startswith(".")}
    print("Number of data", len(simplefold_datas_names))

    common = set(simplefold_datas_names.keys())

    # Create a dataframe with the following schema:
    # tool, name, metric, oracle, average, top1
    results = []
    for name in tqdm(common):
        try:
            simplefold_results = compute_metrics(
                simplefold_datas_names[name],
                simplefold_evals,
                name,
            )
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

        for metric_name in simplefold_results:
            results.append({
                "tool": "SimpleFold",
                "target": name,
                "metric": metric_name,
                "value": simplefold_results[metric_name]["res"],
            })

    # Write the results to a file, ensure we only keep the target & metrics where we have all tools
    df = pd.DataFrame(results)
    df.to_csv(simplefold_evals.parents[0] / "results.csv", index=False)

    # Apply bootstrap to each (tool, metric) group
    boot_stats = df.groupby(["tool", "metric"])["value"].apply(bootstrap_ci)

    # boot_stats is a Series of tuples (mean, lower, upper). Convert to DataFrame:
    boot_stats = boot_stats.apply(pd.Series)
    boot_stats.columns = ["mean", "lower", "upper", "median"]

    # Unstack to get a DataFrame suitable for plotting
    plot_data = boot_stats['mean'].unstack('tool')
    plot_data_median = boot_stats['median'].unstack('tool')

    # plot_data = plot_data.rename(index=renaming)
    print("---------------------- Mean Results ----------------------")
    print(f"TM score: {plot_data.loc['tm_score', 'SimpleFold']:0.4f}")
    print(f"GDT-TS  : {plot_data.loc['oligo_gdtts', 'SimpleFold']:0.4f}")
    print(f"LDDT    : {plot_data.loc['lddt', 'SimpleFold']:0.4f}")
    print(f"BB-LDDT : {plot_data.loc['bb_lddt', 'SimpleFold']:0.4f}")
    print(f"RMSD    : {plot_data.loc['rmsd', 'SimpleFold']:0.4f}")

    plot_data_median = boot_stats['median'].unstack('tool')
    # plot_data_median = plot_data_median.rename(index=renaming)
    print("---------------------- Median Results ----------------------")
    print(f"TM score: {plot_data_median.loc['tm_score', 'SimpleFold']:0.4f}")
    print(f"GDT-TS  : {plot_data_median.loc['oligo_gdtts', 'SimpleFold']:0.4f}")
    print(f"LDDT    : {plot_data_median.loc['lddt', 'SimpleFold']:0.4f}")
    print(f"BB-LDDT : {plot_data_median.loc['bb_lddt', 'SimpleFold']:0.4f}")
    print(f"RMSD    : {plot_data_median.loc['rmsd', 'SimpleFold']:0.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the PDB mmcif directory")
    parser.add_argument("--sample_dir", type=str, required=True, help="Path to the sampled mmcif directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()

    args.data = Path(args.data_dir)
    args.sample = Path(args.sample_dir)
    args.outdir = Path(args.out_dir)

    run_eval(args)
    aggregate_eval(args)