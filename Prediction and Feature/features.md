# Feature set

Total model features: 72 (incl. region_encoded)

## New survival / cadence features added this iteration

- **active_span_days** (Cadence)
- **gap_mean** (Cadence)
- **activity_coverage** (Cadence)
- **gap_max** (Cadence)
- **gap_median** (Cadence)
- **gap_cv** (Cadence)
- **recency_over_gap_mean** (Survival)
- **any_recency_over_gap_mean** (Survival)
- **recency_minus_gap_max** (Survival)
- **unprecedented_silence** (Survival)
- **churn_prob_poisson_out** (Survival)
- **churn_prob_poisson_any** (Survival)
- **days_since_last_any_activity** (Multi-source recency)
- **any_active_last_3d** (Multi-source recency)
- **total_activity_30d** (Multi-source recency)
- **total_activity_90d** (Multi-source recency)
- **days_since_balance_change** (Balance-panel activity)
- **balance_change_count_90d** (Balance-panel activity)
- **balance_change_gap_mean** (Balance-panel activity)