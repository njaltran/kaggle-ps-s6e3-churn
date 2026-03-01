# Run s001 Notes

Model: `lgbm` | Features: `v0`

## CV result
- OOF AUC: **0.916016**
- Fold AUCs: [0.915586, 0.916888, 0.916007, 0.917103, 0.914546]
- Std: 0.000926

## LB result
- Public LB AUC: **(record after submission)**

## Conclusion
- (fill in)

## Next steps
- (fill in)

## LB Result (recorded 2026-03-01)
- **Public LB AUC: 0.91368**
- **Rank: #1 / 3 teams**
- CV-LB gap: 0.0023 (OOF 0.9160 vs LB 0.9137) — excellent calibration
- Baseline beaten: 0.90504 → 0.91368 (+0.0086)

## Conclusion
Outstanding baseline. LightGBM with default params on label-encoded features achieves
rank #1. The synthetic Telco churn data is well-suited to GBDT. CV is reliable.

## Next steps (s002)
- Add feature engineering v1: charges_ratio, avg_monthly_charge, n_addon_services,
  is_month_to_month, has_fiber, high_risk_flag, tenure_bin, monthly_charge_bin
- Expected lift: +0.003–0.005 OOF AUC
