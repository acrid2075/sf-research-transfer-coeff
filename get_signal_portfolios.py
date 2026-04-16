import os
import ray
os.environ["POLARS_MAX_THREADS"] = "1"
import polars as pl
import numpy as np
import datetime as dt
from ray.experimental import tqdm_ray
import sf_quant.data as sfd
import sf_quant.backtester as sfb
import sf_quant.performance as sfp
import sf_quant.optimizer as sfo
import matplotlib.pyplot as plt
from sf_quant.data.benchmark import load_benchmark
from sf_quant.data.covariance_matrix import construct_factor_model_components
from sf_quant.optimizer.optimizers import dynamic_mve_optimizer
from sf_quant.optimizer.constraints import Constraint
from sf_quant.schema.portfolio_schema import PortfolioSchema
import time

# Empty runtime_env = no environment management. Workers use parent's Python directly.
# This prevents Ray from trying to serialize code or rebuild packages on air-gapped clusters.

def connect_to_ray(min_nodes: int):
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    for i in range(30):
        alive = [n for n in ray.nodes() if n.get("Alive", False)]
        print(f"Ray nodes alive: {len(alive)}/{min_nodes}")
        if len(alive) >= min_nodes:
            resources = ray.cluster_resources()
            print(f"Connected. Cluster resources: {resources}")
            return
        time.sleep(5)

    raise RuntimeError(f"Timed out waiting for {min_nodes} Ray nodes.")

# Execution
connect_to_ray(min_nodes=4)
resources = ray.cluster_resources()
print(f"Ray cluster is online with {resources.get('CPU')} total CPUs available.")


@ray.remote(num_cpus=1, max_retries=1)
def _construct_portfolio_for_date(
    date_: dt.date,
    date_data: pl.DataFrame,
    constraints: list[Constraint],
    gamma: float,
    target_active_risk: float | None = None,
    active_weights: bool = False,
):
    subset = date_data.sort("barrid")
    barrids = subset["barrid"].to_list()
    alphas = subset["alpha"].to_numpy()

    betas = (
        subset["predicted_beta"].to_numpy()
        if "predicted_beta" in subset.columns
        else None
    )

    benchmark_weights = (
        subset["benchmark_weight"].to_numpy()
        if "benchmark_weight" in subset.columns
        else None
    )

    if benchmark_weights is None and not active_weights:
        raise ValueError(
            "Benchmark weights must be provided if active_weights=False."
        )

    factor_exposures, factor_covariance, specific_risk = \
        construct_factor_model_components(date_, barrids)

    portfolio = dynamic_mve_optimizer(
        ids=barrids,
        alphas=alphas,
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        constraints=constraints,
        initial_gamma=gamma,
        betas=betas,
        target_active_risk=target_active_risk,
        benchmark_weights=benchmark_weights,
        active_weights=active_weights,
    )

    return portfolio.with_columns(
        pl.lit(date_).alias("date")
    ).select("date", "barrid", "weight", "gamma", "active_risk")


def dynamic_backtest_parallel(
    data: pl.DataFrame,
    constraints: list[Constraint],
    initial_gamma: float = 100,
    target_active_risk: float = 0.05,
    active_weights: bool = False,
    max_in_flight: int = 128,
) -> pl.DataFrame:
    dates = data["date"].unique().sort().to_list()
    date_slices = {
        date_: data.filter(pl.col("date") == date_)
        for date_ in dates
    }

    pending = []
    results = []
    date_iter = iter(dates)

    for _ in range(min(max_in_flight, len(dates))):
        d = next(date_iter, None)
        if d is None:
            break
        pending.append(
            _construct_portfolio_for_date.remote(
                date_=d,
                date_data=ray.put(date_slices[d]),
                constraints=constraints,
                gamma=initial_gamma,
                target_active_risk=target_active_risk,
                active_weights=active_weights,
            )
        )

    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        results.extend(ray.get(done))

        d = next(date_iter, None)
        if d is not None:
            pending.append(
                _construct_portfolio_for_date.remote(
                    date_=d,
                    date_data=ray.put(date_slices[d]),
                    constraints=constraints,
                    gamma=initial_gamma,
                    target_active_risk=target_active_risk,
                    active_weights=active_weights,
                )
            )

    return pl.concat(results)


alphas = pl.read_parquet("/home/acriddl2/groups/grp_quant/database/production/alphas/alphas.parquet")
signal_names = ["barra_momentum", "ivol", "barra_reversal", "reversal", "momentum", "beta"]
zero_constraints = [sfo.ZeroBeta(), sfo.ZeroInvestment()]
unit_constraints = [sfo.UnitBeta(), sfo.LongOnly(), sfo.FullInvestment()]
asset_data = sfd.load_assets(alphas.select(pl.col("date")).min().item(), alphas.select(pl.col("date")).max().item(), columns=["date", "barrid", "predicted_beta", "specific_risk", "return"])

bmk = load_benchmark(
    alphas.select(pl.col("date")).min().item(),
    alphas.select(pl.col("date")).max().item(),
).rename({"weight": "benchmark_weight"})

prepared_data = (
    alphas
    .join(asset_data, on=["date", "barrid"], how="inner")
    .join(bmk, on=["date", "barrid"], how="left")
    .with_columns(pl.col("benchmark_weight").fill_null(0.0))
)

output_dir = "signal_a_w"
os.makedirs(output_dir, exist_ok=True)

for signal in signal_names:
    start = time.time()
    print(f"Processing signal: {signal}")
    signal_data = prepared_data.filter(pl.col("signal_name") == signal)
    # signal_data_ref = ray.put(signal_data)
    
    if signal_data.is_empty():
        print(f"Skipping {signal}: No data found.")
        continue
        
    signal_active_weights = dynamic_backtest_parallel(
        signal_data, 
        zero_constraints, 
        active_weights=True
    )
    signal_active_weights.write_parquet(f"{output_dir}/{signal}_a_w.parquet")
    print(f"  Finished in {time.time() - start}")

output_dir = "signal_t_w"
os.makedirs(output_dir, exist_ok=True)

for signal in signal_names:
    start = time.time()
    print(f"Processing signal: {signal}")
    signal_data = prepared_data.filter(pl.col("signal_name") == signal)
    # signal_data_ref = ray.put(signal_data)
    
    if signal_data.is_empty():
        print(f"Skipping {signal}: No data found.")
        continue
        
    signal_total_weights = dynamic_backtest_parallel(
        signal_data, 
        unit_constraints, 
        active_weights=False
    )
    
    signal_total_weights.write_parquet(f"{output_dir}/{signal}_t_w.parquet")
    print(f"  Finished in {time.time() - start}")


asset_returns = sfd.load_assets(alphas.select(pl.col("date")).min().item(), alphas.select(pl.col("date")).max().item(), columns=["date", "barrid", "predicted_beta", "specific_risk", "return"])
bmk = load_benchmark(alphas.select(pl.col("date")).min().item(), alphas.select(pl.col("date")).max().item()).rename({"weight":"bmk"})
for signal in signal_names:
    print(f"Analyzing signal {signal}")

    signal_active_weights = pl.read_parquet(f"signal_a_w/{signal}_a_w.parquet").rename({"weight":"active_weight"})
    signal_total_weights = pl.read_parquet(f"signal_t_w/{signal}_t_w.parquet").rename({"weight":"total_weight"})
    signal_data = (
        bmk
        .join(
            signal_active_weights,
            on=["date", "barrid"],
            how="right",
        )
        .join(
            signal_total_weights,
            on=["date", "barrid"],
            how="full",
            coalesce=True,
        )
        .with_columns([
            pl.col("bmk").fill_null(0.0),
            pl.col("active_weight").fill_null(0.0),
            pl.col("total_weight").fill_null(0.0),
        ])
        .with_columns(
            (pl.col("total_weight") - pl.col("bmk")).alias("effective_active_weight")
        )
        .join(
            asset_returns,
            on=["date", "barrid"],
            how="inner"
        )
        .sort(["date", "barrid"])
    )

    signal_returns = (signal_data
                   .sort(["barrid", "date"])
                   .with_columns(pl.col("return").truediv(100).shift(-2).over("barrid").alias("fwd_return")
                                 )
                   .with_columns(pl.col("fwd_return").mul(pl.col("active_weight")).alias("active_return"))
                   .group_by("date")
                   .agg(pl.col("active_return").sum().alias("active_return"))
                   .select(["date", "active_return"])
                   .drop_nulls()
                   .sort("date")
                   .with_columns(pl.col("active_return"))
                   .select("date", "active_return")
                   )
    
    tc_by_date = (
        signal_data
        .group_by("date")
        .agg(pl.corr("active_weight", "effective_active_weight").alias("tc"))
        .sort("date")
    )

    curr_vol = signal_returns.select("active_return").std().item()
    print(f"  Current annual active risk: {curr_vol * np.sqrt(252)}")

    long_only_lost_weight = (
        signal_data.with_columns(
            (pl.col("active_weight").truediv(curr_vol).mul(.00315) + pl.col("bmk")).alias("preferred_total")
            )
            .group_by("date").agg(pl.col("preferred_total").filter(pl.col("preferred_total") < 0).abs().sum().truediv(pl.col("active_weight").abs().sum().truediv(2)).alias("lostfrac"))
    )

    scatter_data = signal_data.filter(pl.col("date").eq(tc_by_date.filter(pl.col("tc").eq(pl.col("tc").min().item())).select("date").max().item()))
    plt.clf()
    plt.scatter(scatter_data.select("active_weight"), scatter_data.select("total_weight"), label="Total Weight")
    plt.title(f"{signal}, Total Weight by Active Weight")
    plt.xlabel("Active Weight")
    plt.ylabel("Total Weight")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{signal}_scatter1.png", dpi=400)

    # scatter_data = signal_data.filter(pl.col("date").eq(tc_by_date.filter(pl.col("tc").eq(pl.col("tc").min().item())).select("date").max().item()))
    plt.clf()
    plt.scatter(scatter_data.select("active_weight"), scatter_data.select("effective_active_weight"), label="Effective Active Weight")
    plt.title(f"{signal}, Effective Active Weight by Active Weight")
    plt.xlabel("Active Weight")
    plt.ylabel("Effective Active Weight")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{signal}_scatter2.png", dpi=400)

    plt.clf()
    plt.scatter(scatter_data.select("bmk"), scatter_data.select("effective_active_weight"), label="Effective Active Weight")
    plt.title(f"{signal}, Effective Active Weight by Benchmark Weight")
    plt.xlabel("Benchmark Weight")
    plt.ylabel("Effective Active Weight")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{signal}_scatter3.png", dpi=400)

    plt.clf()
    plt.plot(tc_by_date["date"].to_list(), tc_by_date["tc"].to_list())
    plt.title(f"{signal} Transfer Coefficient")
    plt.xlabel("Time")
    plt.ylabel("Correlation between Active and Total - BMK")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{signal}_tc.png", dpi=400)

    plt.clf()
    plt.plot(long_only_lost_weight["date"].to_list(), long_only_lost_weight["lostfrac"].to_list())
    plt.title(f"{signal}, Active Short Weight Missing from Long Only")
    plt.xlabel("Time")
    plt.ylabel("Fraction Active Short Missing due to Long Only")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"figures/{signal}_lf.png", dpi=400)