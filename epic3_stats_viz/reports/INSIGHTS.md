# Business Insights Report

*Generated 2026-09-27 - Epic 3 checkpoint, Day 15*

## Key Findings

1. Orders average $270.20 per unit (median $284.65), with a standard deviation of $136.85 - prices are widely spread across a range of $491.87, not clustered around one typical value.
2. Typical order quantity is 5.2 units (mode 6), with an IQR of 4.75 units - most orders fall within a fairly narrow band.
3. The strongest pairwise relationship found is between quantity and standard_discount_pct (Pearson r = -0.092), and even that is weak. The weakest is between quantity and unit_price (r = -0.021). No numeric column pair shows a meaningful linear relationship in this dataset.
4. Shapiro-Wilk testing rejects normality (p < 0.05) for quantity and unit_price - downstream analysis should not assume a bell-curve shape for these columns.
5. Z-score outlier screening (±2σ) flagged 0 extreme value(s) across quantity, unit_price, standard_discount_pct - pricing and quantities are consistent, with no runaway values distorting the averages.
6. Clothing is the highest-volume category with 42 orders, versus only 13 for Home & Kitchen - a 3.2x difference in order volume between the busiest and quietest category.
7. Sports has the highest median unit_price ($309.12) among categories, while Books has the lowest ($263.10) - a $46.02 gap that the boxplot/violin charts make visible at a glance.

## Recommendations

1. Because no numeric column pair shows a meaningful linear (or monotonic) relationship, do not rely on quantity, price, or discount alone to predict one another - collect additional features (customer segment, marketing channel, seasonality) before attempting predictive modeling on this data.
2. Since quantity and unit_price both fail normality testing, avoid statistical methods that assume a normal distribution (e.g. raw z-tests or Pearson-based significance tests without checking assumptions) - prefer non-parametric methods (Spearman correlation, rank-based tests) or transform the data first.
3. Investigate why Home & Kitchen has such a low order volume relative to Clothing (13 vs. 42 orders) - determine whether this reflects genuine low demand or a fixable gap in marketing/inventory support before deprioritizing the category.
