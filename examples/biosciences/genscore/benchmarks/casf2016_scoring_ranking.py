import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from _common import add_model_args, formatter, score_preprocessed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GenScore CASF-2016 scoring/ranking benchmark.",
        formatter_class=formatter(),
    )
    add_model_args(parser)
    parser.add_argument("--data-dir", required=True, help="Directory containing preprocessed CASF tensors.")
    parser.add_argument("--test-prefix", default="v2020_casf")
    parser.add_argument("--coreset-file", required=True, help="Path to CASF-2016 CoreSet.dat.")
    parser.add_argument("--outdir", required=True, help="Output directory for CASF ranking .dat file.")
    return parser.parse_args()


def obtain_metrics(df):
    regr = linear_model.LinearRegression()
    regr.fit(df.score.values.reshape(-1, 1), df.logKa.values.reshape(-1, 1))
    preds = regr.predict(df.score.values.reshape(-1, 1))
    rp = pearsonr(df.logKa, df.score)[0]
    mse = mean_squared_error(df.logKa, preds)
    num = df.shape[0]
    sd = np.sqrt((mse * num) / (num - 1))
    print("The regression equation: logKa = %.2f + %.2f * Score" % (float(regr.coef_), float(regr.intercept_)))
    print("Number of favorable sample (N): %d" % num)
    print("Pearson correlation coefficient (R): %.3f" % rp)
    print("Standard deviation in fitting (SD): %.2f" % sd)


def main():
    args = parse_args()
    prots = os.path.join(args.data_dir, f"{args.test_prefix}_prot.pt")
    ligs = os.path.join(args.data_dir, f"{args.test_prefix}_lig.pt")
    ids = os.path.join(args.data_dir, f"{args.test_prefix}_ids.npy")

    _, preds = score_preprocessed(ids, prots, ligs, args)

    core = pd.read_csv(args.coreset_file, sep=r"[,,\t, ]+", header=0, engine="python")
    df_score = pd.DataFrame(zip(np.load(ids, allow_pickle=True)[0], preds), columns=["#code", "score"])
    testdf = pd.merge(core, df_score, on="#code")

    os.makedirs(args.outdir, exist_ok=True)
    testdf[["#code", "score"]].to_csv(
        os.path.join(args.outdir, f"{args.outprefix}.dat"),
        index=False,
        sep="\t",
    )
    obtain_metrics(testdf)


if __name__ == "__main__":
    main()
