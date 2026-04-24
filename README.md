---

# 🧠 Macro Absolute Alpha

### Systematic Long/Short Strategy Driven by Macroeconomic Regimes

---

## 📌 Investment Thesis

Financial markets are not stationary — they evolve through **distinct macroeconomic regimes** driven by shifts in growth, inflation, liquidity, and financial conditions.

Traditional strategies fail because they assume:

* static correlations
* stable factor premia
* constant risk dynamics

This framework is built on a different premise:

> **Alpha emerges from adapting to regime shifts and exploiting cross-sectional dispersion.**

---

## ⚙️ Strategy Overview

This project implements a **systematic long/short equity strategy** that dynamically adjusts exposure based on macroeconomic conditions.

The architecture consists of four layers:

---

### 1. 🧭 Regime Detection (Macro Layer)

* Model: **Hidden Markov Model (HMM)**
* Objective: Identify latent macroeconomic states

Inputs:

* Growth indicators
* Inflation dynamics
* Liquidity conditions
* Financial stress proxies

The model classifies the market into **distinct regimes**, such as:

* Expansion
* Tightening
* Disinflation
* Stress / Crisis

---

### 2. 📊 Cross-Sectional Alpha (Industry Selection)

Within each regime:

* Universe: **49 U.S. industries**
* Ranking based on:

  * **Sortino Ratio** → downside-adjusted performance
  * **Tail Ratio** → asymmetry of returns

Portfolio construction:

* **Long Book** → structurally resilient industries
* **Short Book** → fragile, underperforming industries

This captures:

* dispersion across sectors
* regime-specific winners and losers

---

### 3. 📉 Trend Alignment (Timing Layer)

A trend filter ensures:

* alignment with **prevailing market direction**
* avoidance of counter-trend positioning

This reduces:

* whipsaw risk
* premature short exposure in bull regimes

---

### 4. ⚖️ Portfolio Optimization (Execution Layer)

Implemented using **CVXPY (convex optimization)**

Objective:

* Maximize risk-adjusted returns
* Control downside risk

Constraints:

* Gross exposure limits
* Long/short balance
* Diversification (position caps)
* Volatility control

Key idea:

> The optimizer does not predict returns — it **allocates risk efficiently**.

---

## 📈 Performance Characteristics

The strategy is designed to:

* Generate **consistent alpha across cycles**
* Maintain **low beta to the S&P 500**
* Reduce **drawdowns during crisis periods**
* Exploit **dispersion during regime transitions**

---

## 📊 Analytics & Reporting

The project includes a full performance analytics pipeline:

* CAGR, Sharpe, Sortino
* Max Drawdown
* Kelly Criterion
* Rolling risk metrics
* Volatility analysis
* Drawdown decomposition
* Monthly return heatmaps
* Return distributions

Benchmark:

* **S&P 500**

---

## 🖥️ Interactive Dashboard

A full **Streamlit dashboard** is included for visualization.

### Features

* Cumulative performance (linear & log scale)
* Rolling volatility and risk metrics
* Drawdown (underwater) analysis
* Monthly return heatmaps
* Distribution of returns
* QuantStats performance tables

### Run locally

```bash
streamlit run performance_stats.py
```

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/macro-absolute-alpha.git
cd macro-absolute-alpha
pip install -r requirements.txt
```

---

## 📁 Project Structure

```bash
.
├── performance_stats.py        # Streamlit dashboard
├── strategy_returns.csv       # Strategy returns
├── requirements.txt
└── README.md
```

---

## 🧠 Key Insights

* **Regime > Factor**
  Static factor investing underperforms when macro regimes shift

* **Dispersion = Opportunity**
  Cross-sectional differences drive long/short alpha

* **Risk Allocation > Return Prediction**
  Portfolio construction is the core edge

* **Crisis = Convexity Opportunity**
  The strategy benefits from volatility spikes and dislocations

---

## ⚠️ Limitations

* Regime models can lag during rapid transitions
* Monthly rebalancing reduces responsiveness
* No transaction cost modeling (yet)
* Performance depends on stability of macro signals

---

## 🔮 Future Development

* Add **weekly / adaptive rebalancing**
* Introduce **options overlay (long convexity / tail hedging)**
* Use **probabilistic regime weighting** instead of discrete states
* Integrate **transaction cost & slippage modeling**
* Expand to **multi-asset allocation (FX, rates, commodities)**

---

## 👤 Author

**Gonzalo Abduca**
🔗 [https://www.linkedin.com/in/gonzaloabduca/](https://www.linkedin.com/in/gonzaloabduca/)

---

## ⚠️ Disclaimer

This project is for research purposes only.
It does not constitute investment advice or an offer to manage capital.
