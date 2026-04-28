# Housing Details V1 Runbook

## Purpose

`Housing economics details` is an estimator-first Streamlit tab for BTO, resale, EC, and private scenarios.  
It computes affordability and exit economics using transparent assumptions and formula outputs.

## What It Covers

- Timeline and occupancy:
  - MOP countdown
  - earliest legal sale timing from purchase
- Grants and return-to-government:
  - grant checkbox selection
  - estimated grant return and levy
  - additional manual government return fields
- Loan and instalments:
  - HDB fixed track or bank tracks
  - monthly schedule with interest, principal, CPF used, cash used
- Cost ledger:
  - upfront, recurring, interest, one-off, exit totals
  - COV/premium and renovation costs
- Profit and proceeds:
  - gross sale proceeds
  - agent fee and loan redemption
  - net proceeds and estimated profit
- Repricing/refinancing:
  - switch month and target rate simulation
  - admin/legal/valuation/clawback fees
  - lock-in prepayment penalty modeling
  - gross vs net savings
- Scenario comparison:
  - Scenario A vs Scenario B overlay with adjustable deltas
  - grouped comparison chart for profit, proceeds, and ownership cost
  - monthly net cash outflow overlay chart

## Assumptions Source

Policy defaults are defined in:

- `configs/housing_finance_v1.yaml`

This file stores:

- effective/update date metadata
- default rates (HDB, bank fixed, bank floating base)
- bank default path for common market structure: `24-month fixed -> SORA + spread`
- grant defaults by household profile
- levy defaults by housing type
- fee defaults (BSD/ABSD baseline rates, legal/valuation, maintenance)

## Formula Reference

- Monthly instalment:
  - amortization based on principal, annual rate, and remaining term
- Total cost of ownership:
  - `upfront_total + recurring_total + interest_total + one_off_total + exit_total`
- Estimated total return to government:
  - `estimated_grant_return + resale_levy + other_government_return`
- Net proceeds after obligations:
  - `expected_sale_price - sale_agent_fee - loan_redemption - total_government_return`
- Estimated profit:
  - `net_proceeds_after_obligations - user_total_contributions`
- Repricing gross interest savings:
  - `baseline_interest_total - repriced_interest_total`
- Repricing net savings:
  - `gross_interest_savings - (admin_fee + legal_fee + valuation_fee + clawback_fee + early_prepayment_penalty)`

## Inputs Checklist

Minimum required:

- housing type and household profile
- purchase price and expected sale price
- years to sell, MOP years, wait years to keys
- loan type, tenure, downpayment, interest rate
- monthly CPF OA available

Optional but recommended:

- grants selected
- levy and other government return override
- COV/premium, renovation
- taxes and fees
- monthly rental inflow (only when policy-legal)

## Verification Steps

1. Run unit tests:

```bash
pytest tests/test_housing_finance.py
```

2. Start app:

```bash
streamlit run streamlit_app.py
```

3. In app:
   - open `Housing economics details`
   - run at least three scenarios:
     - SG+SG BTO
     - SG+PR resale
     - EC or private with bank loan
   - confirm tables render:
     - Return to Government
     - All Costs Itemized
     - Profit Breakdown
     - Instalment Cash Flow

## Limitations

- Estimator only, not legal/financial advice.
- Defaults may drift when policy changes; verify with official Singapore sources.
- Grant return and levy are modeled assumptions unless exact case rules are provided.
- Profit output is scenario-dependent and sensitive to sale-price assumption.
- Bank packages vary by institution and repricing date; treat 2-year fixed then SORA as a practical default, not a guarantee.
- Fee and lock-in terms are package-specific; always check your current letter of offer/supplemental offer.

## Online references used for assumptions

- MAS SORA benchmark publication and compounded SORA background:
  - https://www.mas.gov.sg/monetary-policy/sora
- OCBC repricing page (processing fee, no legal/valuation for repricing, lock-in and fee considerations):
  - https://www.ocbc.com/personal-banking/loans/home-loan-repricing
- DBS repricing FAQ (prepayment charges/clawback can apply within commitment period):
  - https://www.dbs.com.sg/personal/loans/homeloans/repricing
