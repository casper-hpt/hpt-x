# Video Content Backlog

## Concepts

### [X] 1. Rebalancing Frequency (Daily vs. Monthly vs. Threshold Drift)
- **Visual Concept:** Run three identical portfolios side-by-side using a dynamic line chart or animated allocation stack. Show how passive drift changes your asset split over time.
- **Educational Hook:** Rebalancing too frequently burns money on trading costs/taxes; rebalancing too rarely ruins risk management because winners take over the portfolio.
- **Key Motion Graphic:** A live "Cost vs. Drift" meter that ticks up as rebalance thresholds are breached, showing the trade-off between keeping targeted weights and incurring friction costs.

---

### [X] 2. Risk Parity vs. Equal Weight vs. Market Cap Weight
- **Visual Concept:** Display 3 distinct asset classes (e.g., Tech Stocks, Crypto, Gold/Bonds). Show a bar chart representing asset weights versus a second set of bars showing Risk Contribution (volatility contribution).
- **Educational Hook:** In an equal-weight portfolio (e.g., 33% each), Crypto might drive 85% of the overall portfolio's variance.
- **Key Motion Graphic:** An animated balance scale or dual-bar chart showing equal capital allocated versus unequal risk, then transitioning to Risk Parity where risk contributions smooth out into flat, equal blocks.

---

### [X] 3. Mean-Variance Optimization: Lookback Window Sensitivity
- **Visual Concept:** Plot an Efficient Frontier curve that visually shifts and morphs in real time as your backtesting clock steps through history (e.g., comparing 30-day, 1-year, and 5-year lookback windows).
- **Educational Hook:** Optimal weights are notoriously sensitive to past data. Small shifts in historical lookbacks lead to radical changes in asset weights ("error maximization").
- **Key Motion Graphic:** An animated heatmap showing asset weight allocations flipping wildly frame-by-frame when using short lookback periods versus stabilizing over longer horizons.

---

### [ ] 4. Correlation Breakdown During Market Crashes
- **Visual Concept:** An animated heatmap matrix of asset correlations (e.g., S&P 500, Real Estate, Emerging Markets, Bonds) during normal market conditions versus a severe stress event (like 2008 or 2020).
- **Educational Hook:** "In a crisis, all correlations go to 1." Diversification appears to vanish right when you need it most.
- **Key Motion Graphic:** Asset returns plotted as drifting nodes in a scatter plot during normal regimes, then suddenly clustering tightly together and falling in unison as the background turns red.

---

### [ ] 5. Execution Lag & Signal Latency
- **Visual Concept:** Split the screen into two equity curves: Executing on the exact close signal vs. executing on the open/close with a 1-bar or multi-bar execution delay.
- **Educational Hook:** Showing how alpha rapidly decays between signal generation and order execution, demonstrating model sensitivity to real-world infrastructure constraints.
- **Key Motion Graphic:** A sliding time cursor highlighting price slippage over time gaps, with the cumulative equity curves diverging wider as time progresses.

---

### [ ] 6. Volatility Targeting (Dynamic Cash Management)
- **Visual Concept:** A two-panel dashboard showing the underlying asset's rolling volatility on top and the portfolio's cash vs. risky asset position on the bottom.
- **Educational Hook:** Scaling position sizes down when volatility spikes keeps portfolio drawdown smooth and avoids sharp market tail-events.
- **Key Motion Graphic:** A dynamic line overlay on the equity curve comparing a static buy-and-hold approach against a smooth, low-drawdown "Vol-Targeted" curve.

---

## Suggested Video Progression

1. **Equal Weight vs. Optimal Weight** *(Current Video)*
2. **Rebalancing Frequency** *(How to maintain those weights in production)*
3. **Risk Parity** *(Weighting by risk rather than capital)*
4. **Correlation Breakdown in Crashes** *(What happens to those weights in a shock)*