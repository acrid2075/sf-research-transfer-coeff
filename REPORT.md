# Research Report

**Project Title: Transfer Coefficient, Analytical Measurements**  
**Author(s): Andrew Criddle, Joseph Moore**  
**Date: 4/11/26**  
**Version: 1.0**  

---

## 1. Summary

Given performance of recently backtested results, we have been concerned over whether our optimizer is correctly implementing our strategies. One possibility is that our constraints are binding our strategies too strictly in a direction incompatible with the economic intuition of those strategies; another is that our optimizer is incorrect.

We find that the transfer coefficient on the active portfolio for each signal is extremely low; we also find that in most implemented signals less than half of our relative short positions in the active portfolio by weight are implemented due to long-only constraints.

## 2. Data Requirements

Describe data dependencies.

**Sources**
-  grp_quant/database/production/alphas/alphas.parquet  # Only reads from, DON'T WRITE TO
-  existing security details from barra
 

---

## 3. Approach / System Design

We consider the standard total constraints long-only, full investment, and unit beta, with a target active risk of .05, and standard active constraints zero beta, zero investment, with a target active risk of .05. For the optimizer, we perform $w^T \mu - \frac{\gamma}{2}w^T \Sigma w$, find the optimal $w$ portfolio weights, and alter gamma until the desired active risk is achieved.

For each signal:

We run the backtest twice, the first time to produce the optimal active portfolio and second time to produce the optimal constrained total portfolio. We then subtract the benchmark weights from the optimal constrained total portfolio producting the effective active weights.

We then construct the transfer coefficient on each day by taking the correlation between the active weights and the effective active weights. We then look at the implemented active short percentage by taking the preferred total portfolio (the original active portfolio plus the benchmark), look at the short positions, sum the weights, take the absolute value, and divide it by half of the sum of the absolute value of each active position in the original active portfolio.

---

## 4. Code Structure

```
sf-research-transfer-coeff/
├── figures/  # Contains the figures for each signal
├── signal_a_w/
│   └── <signal>_a_w.parquet  # Active weights for the signal
├── signal_t_w/
│   └── <signal>_t_w.parquet  # Total weights for the signal
├── get_signal_portfolios.py  # Python producing the relevant parquet files and figures
├── get_signal_portfolios.sh  # Allows for leveraging parallel processing for analysis
├── REPORT.md
└── README.md
```

Run the following in the directory:

   sbatch get_signal_portfolios.sh

---

## 5. Results / Evaluation

Include relevant evidence demonstrating performance.

For signals:

- Cumulative IC table
- Possibly quantile plots
- Active portfolio backtest
- Summary statistic tables
- Other useful tables and plots

Possible items:

- Tables
- Plots
- Benchmarks

Add anything useful for interpreting system behavior.

Barra Momentum:
   ![barra_momentum Transfer Coefficient Line Plot](figures/barra_momentum_tc.png)

   ![barra_momentum Long Only Ratio Line Plot](figures/barra_momentum_lf.png)

   ![barra_momentum Effective Weight by Active Weight Scatter Plot](figures/barra_momentum_scatter2.png)

Barra Reversal:
   ![barra_reversal Transfer Coefficient Line Plot](figures/barra_reversal_tc.png)

   ![barra_reversal Long Only Ratio Line Plot](figures/barra_reversal_lf.png)

   ![barra_reversal Effective Weight by Active Weight Scatter Plot](figures/barra_reversal_scatter2.png)

Betting Against Beta:
   ![beta Transfer Coefficient Line Plot](figures/beta_tc.png)

   ![beta Long Only Ratio Line Plot](figures/beta_lf.png)

   ![beta Effective Weight by Active Weight Scatter Plot](figures/beta_scatter2.png)

Idiosyncratic Volatility:
   ![ivol Transfer Coefficient Line Plot](figures/ivol_tc.png)

   ![ivol Long Only Ratio Line Plot](figures/ivol_lf.png)

   ![ivol Effective Weight by Active Weight Scatter Plot](figures/ivol_scatter2.png)

Momentum:
   ![momentum Transfer Coefficient Line Plot](figures/momentum_tc.png)

   ![momentum Long Only Ratio Line Plot](figures/momentum_lf.png)

   ![momentum Effective Weight by Active Weight Scatter Plot](figures/momentum_scatter2.png)

Reversal:
   ![reversal Transfer Coefficient Line Plot](figures/reversal_tc.png)

   ![reversal Long Only Ratio Line Plot](figures/reversal_lf.png)

   ![reversal Effective Weight by Active Weight Scatter Plot](figures/reversal_scatter2.png)

---

## 6. Performance Discussion

We note that the transfer coefficients on every signal is extremely concerning, deserving further analysis. When we combine the various signals, it is entirely possible that the resulting active portfolio will have less significant short positions in each lower-cap stock, allowing for a decreased constraint directly from the long-only implementation. 

However, given how low each transfer coefficient is for every signal, it seems most probable that our optimizer itself is wrong, or that our optimized problem is incorrect, as even if our effective active portfolio was entirely long only we would anticipate approximately a transfer coefficient of .5, and we see much closer to a value of .1 for most signals.

This is extremely concerning, and entirely deserving of further analysis.

---