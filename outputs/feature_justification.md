FEATURE ENGINEERING JUSTIFICATION
=================================

INCLUDED FEATURES (9 deployed)
------------------------------

1. age_education_interaction
   - Reason: Captures combined effect of age and education on income.
   - Impact: Improves discrimination vs using age or education alone.
   - Fairness: Considered safe; no direct sensitive attribute.

2. capital_net
   - Reason: Combines capital_gain and capital_loss into a single net wealth signal.
   - Impact: Handles skewed gains/losses better than raw fields.
   - Fairness: Pure financial metric; low fairness risk.

3. has_capital_gain
   - Reason: Binary indicator for any investment income.
   - Impact: Adds signal even when magnitude is noisy.
   - Fairness: Financial indicator; low risk.

4. has_capital_loss
   - Reason: Binary indicator for any investment loss.
   - Impact: Complements capital_net by highlighting loss presence.
   - Fairness: Financial indicator; low risk.

5. is_overtime
   - Reason: Distinguishes regular vs overtime workers.
   - Impact: Strong predictor of higher income.
   - Fairness: Medium risk (may disadvantage part-time workers). Monitor by gender.

6. education_bucket
   - Reason: Groups many education levels into 4 buckets to reduce sparsity.
   - Impact: Stabilizes estimates and improves generalization.
   - Fairness: Designed to treat education tiers consistently; low risk.

7. is_professional
   - Reason: Flags professional and executive occupations.
   - Impact: Strong predictor of higher income.
   - Fairness: Medium risk due to occupational gender imbalance. Monitor distributions.

8. professional_overtime
   - Reason: Interaction between professional status and overtime.
   - Impact: Captures very strong high-income signal.
   - Fairness: Medium risk; combines two medium-risk features. Monitor carefully.

9. hours_bin
   - Reason: Bins working hours into categories to target false negatives.
   - Impact: Improves recall on high-income class.
   - Fairness: Medium risk for part-time workers; monitor by gender.

EXCLUDED FEATURES (2 high-risk)
-------------------------------

1. is_married
   - Reason: Strong proxy for gender norms; large TPR disparity across groups (1.92x).
   - Decision: Excluded from production for fairness.
   - Performance impact: -0.56% F1 (acceptable trade-off).
   - Plan: Consider A/B testing in the future with strict monitoring.

2. age_married_interaction
   - Reason: Interaction built on a high-risk marital feature.
   - Decision: Excluded together with is_married.
   - Performance impact: Same as is_married.

FINAL MODEL PERFORMANCE
-----------------------
Train AUC: 0.9071, F1: 0.6812
Val AUC: 0.9089, F1: 0.6805
Test AUC: 0.9075, F1: 0.6802

Status: ✅ REPRODUCIBLE (identical across multiple runs with random_state=42)
