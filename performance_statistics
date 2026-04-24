import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import quantstats as qs
import matplotlib as mpl
import matplotlib.ticker as mtick
import streamlit as st
from contextlib import redirect_stdout
import io


def set_dark_finance_theme():
    mpl.rcParams.update({
        "figure.facecolor": "#121212",
        "axes.facecolor": "#121212",
        "savefig.facecolor": "#121212",
        "savefig.edgecolor": "#121212",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "text.color": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "#3a3a3a",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.grid": True,
        "legend.facecolor": "#121212",
        "legend.edgecolor": "white",
        "legend.framealpha": 0.9,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

set_dark_finance_theme()

STRATEGY_COLOR = "#00E5FF"   # cyan
BENCHMARK_COLOR = "#FF9F1C"  # orange
GRID_COLOR = "#3a3a3a"
BG_COLOR = "#121212"
FG_COLOR = "white"

def style_dark_ax(ax, title=None, xlabel=None, ylabel=None, legend=True):
    
    ax.set_facecolor(BG_COLOR)

    if title is not None:
        ax.set_title(title, color=FG_COLOR, fontsize=14, pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=FG_COLOR)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=FG_COLOR)

    ax.tick_params(axis="x", colors=FG_COLOR)
    ax.tick_params(axis="y", colors=FG_COLOR)

    for spine in ax.spines.values():
        spine.set_color(FG_COLOR)

    ax.grid(True, color=GRID_COLOR, alpha=0.35, linestyle="--")

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(BG_COLOR)
        legend.get_frame().set_edgecolor(FG_COLOR)
        for text in legend.get_texts():
            text.set_color(FG_COLOR)

def plot_cumulative_performance(strategy_curve, benchmark_curve, figsize=(12, 6), legend=True):

    fig, ax = plt.subplots(figsize=figsize)

    strategy_curve.plot(ax=ax, color=STRATEGY_COLOR, linewidth=2.4, label="Strategy")
    benchmark_curve.plot(ax=ax, color=BENCHMARK_COLOR, linewidth=2.0, label="Benchmark")

    style_dark_ax(
        ax,
        title="Cumulative Performance",
        xlabel="Date",
        ylabel="Growth of $1"
    )

    if legend:
        ax.legend()

        # 👉 STYLE legend
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.get_frame().set_facecolor(BG_COLOR)
            legend_obj.get_frame().set_edgecolor(FG_COLOR)
            for text in legend_obj.get_texts():
                text.set_color(FG_COLOR)

    plt.tight_layout()
    return fig, ax

def plot_log_cumulative_returns(strategy_ret, benchmark_ret, figsize=(12, 6), legend=True):
    strategy_curve = (1 + strategy_ret).cumprod() - 1
    benchmark_curve = (1 + benchmark_ret).cumprod() - 1

    strategy_plot = 1 + strategy_curve
    benchmark_plot = 1 + benchmark_curve

    fig, ax = plt.subplots(figsize=figsize)

    benchmark_plot.plot(ax=ax, color=BENCHMARK_COLOR, linewidth=2.0, label="Benchmark")
    strategy_plot.plot(ax=ax, color=STRATEGY_COLOR, linewidth=2.4, label="Strategy")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda y, _: f"{(y - 1):.0%}")
    )
    ax.axhline(1, color="white", linestyle="--", linewidth=1, alpha=0.8)

    style_dark_ax(
        ax,
        title="Cumulative Returns vs Benchmark (Log Scale)",
        xlabel="Date",
        ylabel="Cumulative Return"
    )
    if legend:
        ax.legend()

        # 👉 STYLE legend
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.get_frame().set_facecolor(BG_COLOR)
            legend_obj.get_frame().set_edgecolor(FG_COLOR)
            for text in legend_obj.get_texts():
                text.set_color(FG_COLOR)

    plt.tight_layout()
    return fig, ax

def plot_volatility_matched_cumulative_returns(
    strategy_ret,
    benchmark_ret,
    figsize=(12, 6),
    title="Cumulative Returns vs Benchmark (Volatility Matched)",
    legend=True
):
    # align data
    df = pd.concat(
        [strategy_ret.rename("strategy"), benchmark_ret.rename("benchmark")],
        axis=1
    ).dropna()

    strategy_ret = df["strategy"]
    benchmark_ret = df["benchmark"]

    # volatility match benchmark to strategy
    strat_vol = strategy_ret.std()
    bench_vol = benchmark_ret.std()

    if bench_vol == 0:
        raise ValueError("Benchmark volatility is zero, cannot volatility-match.")

    vol_ratio = strat_vol / bench_vol
    benchmark_matched = benchmark_ret * vol_ratio

    # cumulative returns
    strategy_curve = (1 + strategy_ret).cumprod() - 1
    benchmark_curve = (1 + benchmark_matched).cumprod() - 1

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    benchmark_curve.plot(
        ax=ax,
        color=BENCHMARK_COLOR,
        linewidth=2.0,
        label="Benchmark"
    )
    strategy_curve.plot(
        ax=ax,
        color=STRATEGY_COLOR,
        linewidth=2.4,
        label="Strategy"
    )

    # zero line
    ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.8)

    # percent axis
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    style_dark_ax(
        ax,
        title=title,
        xlabel="Date",
        ylabel="Cumulative Return"
    )
    if legend:
        ax.legend()

        # 👉 STYLE legend
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.get_frame().set_facecolor(BG_COLOR)
            legend_obj.get_frame().set_edgecolor(FG_COLOR)
            for text in legend_obj.get_texts():
                text.set_color(FG_COLOR)

    plt.tight_layout()
    return fig, ax


def plot_eoy_returns_vs_benchmark(
    strategy_ret,
    benchmark_ret,
    figsize=(14, 6),
    title="EOY Returns vs Benchmark",
    reference_line=None,
    strategy_label="Strategy",
    benchmark_label="Benchmark",
    year_step=5,
    legend=True
):
    df = pd.concat(
        [strategy_ret.rename("strategy"), benchmark_ret.rename("benchmark")],
        axis=1
    ).dropna()

    yearly = df.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    yearly.index = yearly.index.year

    x = np.arange(len(yearly))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x - width/2,
        yearly["benchmark"],
        width=width,
        color=BENCHMARK_COLOR,
        alpha=0.9,
        label=benchmark_label
    )

    ax.bar(
        x + width/2,
        yearly["strategy"],
        width=width,
        color=STRATEGY_COLOR,
        alpha=0.95,
        label=strategy_label
    )

    ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.8)

    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="#FF4D4D",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9
        )

    tick_idx = np.arange(0, len(yearly), year_step)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(yearly.index[tick_idx].astype(str), rotation=30, ha="right", color=FG_COLOR)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    style_dark_ax(
        ax,
        title=title,
        xlabel="",
        ylabel="Annual Return"
    )

    if legend:
        ax.legend()

        # 👉 STYLE legend
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.get_frame().set_facecolor(BG_COLOR)
            legend_obj.get_frame().set_edgecolor(FG_COLOR)
            for text in legend_obj.get_texts():
                text.set_color(FG_COLOR)

    plt.tight_layout()
    return fig, ax



def plot_monthly_returns_heatmap(
    returns,
    title="Strategy - Monthly Returns (%)",
    figsize=(14, 12),
    cmap="RdYlGn",
    vmin=None,
    vmax=None,
    fill_missing=True
):
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.dropna()

    # Monthly compounded returns
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Build year x month table in %
    heatmap_df = monthly.to_frame("ret")
    heatmap_df["Year"] = heatmap_df.index.year
    heatmap_df["Month"] = heatmap_df.index.month

    pivot = heatmap_df.pivot(index="Year", columns="Month", values="ret") * 100

    # Ensure all months exist
    pivot = pivot.reindex(columns=range(1, 13))

    if fill_missing:
        pivot = pivot.fillna(0)

    month_labels = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    data = pivot.values

    # Symmetric color scale around zero if not supplied
    if vmin is None or vmax is None:
        max_abs = np.nanmax(np.abs(data))
        vmin = -max_abs if vmin is None else vmin
        vmax =  max_abs if vmax is None else vmax

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    # Axis labels
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(month_labels, color=FG_COLOR, fontsize=11)

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str), color=FG_COLOR, fontsize=11)

    ax.set_title(title, color=FG_COLOR, fontsize=15, pad=12)
    
    ax.grid(False)

    # # Cell borders / grid
    # ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    # ax.set_yticks(np.arange(-.5, len(pivot.index), 1), minor=True)
    # ax.grid(which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0)
    # ax.tick_params(which="minor", bottom=False, left=False)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]

            if np.isnan(val):
                label = ""
            else:
                label = f"{val:.2f}"

            # White text on stronger colors, dark text on pale cells
            text_color = "white" if abs(val) > (0.45 * max(abs(vmin), abs(vmax))) else "#2b2b2b"

            ax.text(
                j, i, label,
                ha="center", va="center",
                color=text_color,
                fontsize=10
            )

    plt.tight_layout()
    return fig, ax


def plot_rolling_volatility(
    strategy_ret,
    benchmark_ret=None,
    window=126,  # ~6 months of trading days
    periods_per_year=252,
    figsize=(12, 6),
    title="Rolling Volatility (6-Months)",
    reference_line=0.10,
    strategy_label="Strategy",
    benchmark_label="Benchmark"
):
    strategy_ret = strategy_ret.copy()
    strategy_ret.index = pd.to_datetime(strategy_ret.index)

    if benchmark_ret is not None:
        benchmark_ret = benchmark_ret.copy()
        benchmark_ret.index = pd.to_datetime(benchmark_ret.index)

        df = pd.concat(
            [strategy_ret.rename("strategy"), benchmark_ret.rename("benchmark")],
            axis=1
        ).dropna()

        strategy_ret = df["strategy"]
        benchmark_ret = df["benchmark"]

    rolling_vol_strategy = (
        strategy_ret.rolling(window).std() * np.sqrt(periods_per_year)
    )

    if benchmark_ret is not None:
        rolling_vol_benchmark = (
            benchmark_ret.rolling(window).std() * np.sqrt(periods_per_year)
        )

    fig, ax = plt.subplots(figsize=figsize)

    rolling_vol_strategy.plot(
        ax=ax,
        color=STRATEGY_COLOR,
        linewidth=2.2,
        label=strategy_label
    )

    if benchmark_ret is not None:
        rolling_vol_benchmark.plot(
            ax=ax,
            color=BENCHMARK_COLOR,
            linewidth=1.9,
            alpha=0.9,
            label=benchmark_label
        )

    # zero line
    ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.8)

    # reference line
    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="#FF4D4D",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9
        )

    style_dark_ax(
        ax,
        title=title,
        xlabel="Date",
        ylabel="Annualized Volatility"
    )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    return fig, ax

def plot_underwater_vs_benchmark(
    strategy_ret,
    benchmark_ret=None,
    figsize=(12, 6),
    title="Underwater Plot vs Benchmark",
    reference_line=None,   # example: -0.08
    strategy_label="Strategy",
    benchmark_label="Benchmark",
    legend=True
):
    strategy_ret = strategy_ret.copy()
    strategy_ret.index = pd.to_datetime(strategy_ret.index)

    if benchmark_ret is not None:
        benchmark_ret = benchmark_ret.copy()
        benchmark_ret.index = pd.to_datetime(benchmark_ret.index)

        df = pd.concat(
            [strategy_ret.rename("strategy"), benchmark_ret.rename("benchmark")],
            axis=1
        ).dropna()

        strategy_ret = df["strategy"]
        benchmark_ret = df["benchmark"]

    # wealth index
    strategy_curve = (1 + strategy_ret).cumprod()
    strategy_dd = strategy_curve / strategy_curve.cummax() - 1

    if benchmark_ret is not None:
        benchmark_curve = (1 + benchmark_ret).cumprod()
        benchmark_dd = benchmark_curve / benchmark_curve.cummax() - 1

    fig, ax = plt.subplots(figsize=figsize)

    # benchmark first so strategy sits on top
    if benchmark_ret is not None:
        ax.fill_between(
            benchmark_dd.index,
            benchmark_dd.values,
            0,
            color=BENCHMARK_COLOR,
            alpha=0.22,
            label=benchmark_label
        )
        ax.plot(
            benchmark_dd.index,
            benchmark_dd.values,
            color=BENCHMARK_COLOR,
            linewidth=1.4,
            alpha=0.9
        )

    ax.fill_between(
        strategy_dd.index,
        strategy_dd.values,
        0,
        color=STRATEGY_COLOR,
        alpha=0.28,
        label=strategy_label
    )
    ax.plot(
        strategy_dd.index,
        strategy_dd.values,
        color=STRATEGY_COLOR,
        linewidth=1.5
    )

    # zero line
    ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.8)

    # optional reference line
    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="#FF4D4D",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9
        )

    style_dark_ax(
        ax,
        title=title,
        xlabel="Date",
        ylabel="Drawdown"
    )

    if legend:
        ax.legend()

        # 👉 STYLE legend
        legend_obj = ax.get_legend()
        if legend_obj is not None:
            legend_obj.get_frame().set_facecolor(BG_COLOR)
            legend_obj.get_frame().set_edgecolor(FG_COLOR)
            for text in legend_obj.get_texts():
                text.set_color(FG_COLOR)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    return fig, ax


def plot_rolling_metric(
    metric_series,
    title="Rolling Sharpe (6-Months)",
    reference_line=1.0,
    figsize=(12, 6),
    label="Strategy"
):
    metric_series = metric_series.copy()
    metric_series.index = pd.to_datetime(metric_series.index)
    metric_series = metric_series.dropna()

    fig, ax = plt.subplots(figsize=figsize)

    metric_series.plot(
        ax=ax,
        color=STRATEGY_COLOR,
        linewidth=2.2,
        label=label
    )

    ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.8)

    if reference_line is not None:
        ax.axhline(
            reference_line,
            color="#FF4D4D",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9
        )

    style_dark_ax(
        ax,
        title=title,
        xlabel="Date",
        ylabel=""
    )

    plt.tight_layout()
    return fig, ax

def plot_monthly_return_distribution(
    strategy_ret,
    benchmark_ret,
    figsize=(12, 6),
    bins=30,
    title="Distribution of Monthly Returns",
    strategy_label="Strategy",
    benchmark_label="S&P 500"
):
    df = pd.concat(
        [
            strategy_ret.rename("strategy"),
            benchmark_ret.rename("benchmark")
        ],
        axis=1
    ).dropna()

    monthly = df.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        monthly["benchmark"],
        bins=bins,
        density=True,
        alpha=0.45,
        color=BENCHMARK_COLOR,
        label=benchmark_label
    )

    ax.hist(
        monthly["strategy"],
        bins=bins,
        density=True,
        alpha=0.45,
        color=STRATEGY_COLOR,
        label=strategy_label
    )

    monthly["benchmark"].plot(
        kind="kde",
        ax=ax,
        color=BENCHMARK_COLOR,
        linewidth=2.0
    )

    monthly["strategy"].plot(
        kind="kde",
        ax=ax,
        color=STRATEGY_COLOR,
        linewidth=2.2
    )

    ax.axvline(0, color="white", linestyle="--", linewidth=1, alpha=0.8)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    style_dark_ax(
        ax,
        title=title,
        xlabel="Monthly Return",
        ylabel="Density"
    )

    ax.legend(loc="upper right")

    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(BG_COLOR)
        legend.get_frame().set_edgecolor(FG_COLOR)
        for text in legend.get_texts():
            text.set_color(FG_COLOR)

    plt.tight_layout()
    return fig, ax

def plot_return_quantiles(
    returns,
    figsize=(12, 6),
    title="Strategy - Return Quantiles",
    label="Strategy"
):
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.dropna()

    quantiles = pd.DataFrame({
        "Daily": returns,
        "Weekly": returns.resample("W").apply(lambda x: (1 + x).prod() - 1),
        "Monthly": returns.resample("ME").apply(lambda x: (1 + x).prod() - 1),
        "Quarterly": returns.resample("QE").apply(lambda x: (1 + x).prod() - 1),
        "Yearly": returns.resample("YE").apply(lambda x: (1 + x).prod() - 1),
    })

    fig, ax = plt.subplots(figsize=figsize)

    box = quantiles.boxplot(
        ax=ax,
        grid=False,
        patch_artist=True,
        showfliers=True,
        return_type="dict"
    )

    colors = ["#8A8A8A", "#00E5FF", "#D85C8A", "#4DB6AC", "#9B59B6"]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(FG_COLOR)

    for element in ["whiskers", "caps", "medians"]:
        for line in box[element]:
            line.set_color(FG_COLOR)
            line.set_linewidth(1.2)

    for flier in box["fliers"]:
        flier.set_markerfacecolor(BG_COLOR)
        flier.set_markeredgecolor(FG_COLOR)
        flier.set_alpha(0.45)

    style_dark_ax(
        ax,
        title=title,
        xlabel="",
        ylabel="Return"
    )

    ax.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    return fig, ax


qs.extend_pandas()

st.set_page_config(
    page_title="Macro Absolute Alpha Report",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #121212;
    color: white;
}
.metric-card {
    background: #1E1E1E;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #333;
}
h1, h2, h3, p, div {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("Macroeconomic Regime Long/Short Asset Allocation")
st.caption("Strategy vs S&P 500 Benchmark")
st.markdown(
    """
    **Trading & Portfolio Management Strategy** engineered by 
    [Gonzalo Abduca](https://www.linkedin.com/in/gonzaloabduca/), 
    using macroeconomic regime identification with machine learning algorithms 
    and CVXPY for portfolio weight optimization.
    """
)

strat_ret = pd.read_csv(
    "C:/Users/User/Desktop/Data Projects/Portfolio Optimisation/1.strategy_returns.csv",
    index_col=0
).squeeze()

strat_ret.index = pd.to_datetime(strat_ret.index)

benchmark = yf.download(
    "^GSPC",
    start=strat_ret.index.min(),
    end=strat_ret.index.max(),
    auto_adjust=True
)["Close"].squeeze().pct_change()

benchmark = benchmark.reindex(strat_ret.index).dropna()
strat_ret = strat_ret.loc[benchmark.index]

cagr = qs.stats.cagr(strat_ret)
sharpe = qs.stats.sharpe(strat_ret)
sortino = qs.stats.sortino(strat_ret)
max_dd = qs.stats.max_drawdown(strat_ret)
kelly = qs.stats.kelly_criterion(strat_ret)
cum_ret = ((1+strat_ret).cumprod() - 1).iloc[-1]

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("CAGR", f"{cagr:.2%}")
col2.metric("Sharpe", f"{sharpe:.2f}")
col3.metric("Sortino", f"{sortino:.2f}")
col4.metric("Max Drawdown", f"{max_dd:.2%}")
col5.metric("Kelly Criterion", f"{kelly:.2%}")
col6.metric("Cummulative Returns", f"{cum_ret:.2%}")


def format_metrics(df):

    percent_rows = [
        "Risk-Free Rate", "Time in Market", "Cumulative Return", "CAGR﹪",
        "Max Drawdown", "Volatility (ann.)", "Avg. Return",
        "Avg. Win", "Avg. Loss", "Expected Daily", "Expected Monthly",
        "Expected Yearly", "Daily Value-at-Risk", "Expected Shortfall (cVaR)",
        "MTD", "3M", "6M", "YTD", "1Y", "3Y (ann.)", "5Y (ann.)",
        "10Y (ann.)", "All-time (ann.)", "Best Day", "Worst Day",
        "Best Month", "Worst Month", "Best Year", "Worst Year",
        "Avg. Drawdown", "Avg. Up Month", "Avg. Down Month",
        "Win Days", "Win Month", "Win Quarter", "Win Year",
        "Kelly Criterion"
    ]

    ratio_rows = [
        "Sharpe", "Sortino", "Calmar", "Omega",
        "Profit Factor", "Tail Ratio", "Gain/Pain Ratio",
        "Gain/Pain (1M)", "Payoff Ratio"
    ]

    prob_rows = [
        "Prob. Sharpe Ratio", "Risk of Ruin"
    ]

    formatted = df.copy()

    for row in formatted.index:
        for col in formatted.columns:
            val = formatted.loc[row, col]

            if pd.isna(val):
                continue

            # Percent rows → 0.57 → 57%
            if row in percent_rows:
                formatted.loc[row, col] = f"{val * 100:.2f}%"

            # Probability rows → 1.0 → 100%
            elif row in prob_rows:
                formatted.loc[row, col] = f"{val * 100:.2f}%"

            # Ratios → keep decimals
            elif row in ratio_rows:
                formatted.loc[row, col] = f"{val:.2f}"

            # Default
            else:
                if isinstance(val, float):
                    formatted.loc[row, col] = f"{val:.2f}"

    return formatted


col_a, col_b = st.columns([1.25, 1])

with col_a:

    st.subheader("Cumulative Performance")

    strategy_curve = (1 + strat_ret).cumprod() - 1
    benchmark_curve = (1 + benchmark).cumprod() - 1

    fig, ax = plot_cumulative_performance(strategy_curve, benchmark_curve, figsize=(12, 8))
    st.pyplot(fig)

    fig, ax = plot_log_cumulative_returns(strat_ret, benchmark, figsize=(12, 8))
    st.pyplot(fig)

    fig, ax = plot_eoy_returns_vs_benchmark(strat_ret, benchmark_ret=benchmark)
    st.pyplot(fig)

    fig, ax = plot_underwater_vs_benchmark(strat_ret, benchmark, figsize=(12, 8))
    st.pyplot(fig)

    fig, ax = plot_rolling_volatility(strat_ret, benchmark, figsize=(12, 8))

    st.pyplot(fig)

    fig, ax = plot_rolling_metric(strat_ret.rolling_sortino(),  figsize=(12, 8), title='Rolling Sortino (6-months)')
    st.pyplot(fig)

    fig, ax = plot_monthly_returns_heatmap(strat_ret, vmin=-10, vmax=10, figsize=(12, 36))
    st.pyplot(fig)

with col_b:
    
    # =========================
    # KEY PERFORMANCE METRICS
    # =========================
    
    st.info("""
    ### Strategy Overview

    A systematic macroeconomic long/short strategy designed to exploit regime shifts and cross-sectional dispersion across equity markets. The framework uses a Hidden Markov Model to identify latent macroeconomic regimes driven by growth, inflation, liquidity, and financial conditions.

    Within each regime, it ranks 49 U.S. industries using downside-adjusted performance metrics such as Sortino and tail ratios to isolate resilient winners and structurally weak losers.

    A trend filter ensures alignment with prevailing market direction, while a convex optimization engine built with CVXPY allocates capital under strict risk constraints, including gross exposure, diversification, and downside risk control.

    The result is a low-beta, highly diversified portfolio that adapts dynamically to changing environments, aiming to deliver consistent alpha, reduce volatility, and limit drawdowns across full market cycles.
    """)

    st.subheader("Key Performance Metrics")
    
    @st.cache_data(ttl=84000)
    def quantstats_metrics_text_to_df(returns, benchmark):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            qs.reports.metrics(
                returns=returns,
                benchmark=benchmark,
                mode="full",
                display=True,
                strategy_title="Macro Absolute Alpha",
                benchmark_title="S&P 500"
            )

        text = buffer.getvalue()
        lines = text.splitlines()

        rows = []
        start = False

        for line in lines:
            if "S&P 500" in line and "Macro Absolute Alpha" in line:
                start = True
                continue

            if not start:
                continue

            if line.strip() == "" or set(line.strip()) <= {"-"}:
                continue

            # split by 2+ spaces
            parts = [p.strip() for p in line.split("  ") if p.strip()]

            if len(parts) >= 3:
                metric = " ".join(parts[:-2])
                benchmark_val = parts[-2]
                strategy_val = parts[-1]

                rows.append([metric, benchmark_val, strategy_val])

        df = pd.DataFrame(
            rows,
            columns=["Metric", "S&P 500", "Macro Absolute Alpha"]
        )

        return df

    metrics_df = quantstats_metrics_text_to_df(strat_ret, benchmark)
    
    st.dataframe(metrics_df, use_container_width=True, height=3000, hide_index=True)

    dd = qs.stats.to_drawdown_series(strat_ret)

    dd_details = qs.stats.drawdown_details(dd)

    worst_10 = (
        dd_details
        .sort_values("max drawdown")   # most negative first
        .head(10)
    )

    st.subheader("Worst Drawdowns Registered")

    st.dataframe(worst_10, use_container_width=True, hide_index=True)

    fig, ax = plot_monthly_return_distribution(
    strat_ret,
    benchmark,
    bins=20,
    benchmark_label="S&P 500",
    strategy_label="Macro Absolute Alpha",
    figsize=(12,10)
    )

    st.pyplot(fig)

    fig, ax = plot_return_quantiles(
    strat_ret,
    title="Strategy - Return Quantiles",
    figsize=(12,10)
    )

    st.pyplot(fig)

    
