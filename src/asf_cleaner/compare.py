"""Compare NISAR vs OPERA trends over their overlap period (informational only)."""
import argparse

import pandas as pd


def compare(nisar: pd.DataFrame, opera: pd.DataFrame, tol_days: int = 6) -> pd.DataFrame:
    out = []
    for cls in ("co", "cross"):
        a = nisar[nisar.pol_class == cls].groupby("start_time").agg(
            n_db=("backscatter_db_median", "mean"), n_z=("z_median", "mean")).reset_index()
        b = opera[opera.pol_class == cls].groupby("start_time").agg(
            o_db=("backscatter_db_median", "mean"), o_z=("z_median", "mean")).reset_index()
        m = pd.merge_asof(a.sort_values("start_time"), b.sort_values("start_time"),
                          on="start_time", tolerance=pd.Timedelta(days=tol_days),
                          direction="nearest").dropna()
        out.append({
            "pol_class": cls,
            "n_pairs": len(m),
            "z_correlation": m["n_z"].corr(m["o_z"]) if len(m) >= 3 else float("nan"),
            "mean_offset_db_nisar_minus_opera": (m["n_db"] - m["o_db"]).mean(),
        })
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare NISAR and OPERA outputs over overlapping dates")
    ap.add_argument("nisar_parquet")
    ap.add_argument("opera_parquet")
    a = ap.parse_args(argv)
    print(compare(pd.read_parquet(a.nisar_parquet), pd.read_parquet(a.opera_parquet)).to_string(index=False))
    print("\nNote: few pairs => correlation is weak evidence. Offsets are NOT applied to the data.")


if __name__ == "__main__":
    main()
