from __future__ import annotations

from pathlib import Path

from singapore_eda.housing_finance.calculators import (
    make_fixed_then_sora_segments,
    run_housing_finance,
)
from singapore_eda.housing_finance.models import (
    GrantSelection,
    HouseholdProfile,
    HousingFinanceScenario,
    HousingType,
    LoanType,
)
from singapore_eda.housing_finance.policy_defaults import load_policy_defaults


def _defaults() -> Path:
    return Path("configs/housing_finance_v1.yaml")


def test_policy_defaults_loads() -> None:
    cfg = load_policy_defaults(_defaults())
    assert cfg.effective_date
    assert cfg.hdb_loan_rate_pct > 0
    assert cfg.bank_fixed_period_months == 24


def test_run_housing_finance_bto_sg_sg() -> None:
    cfg = load_policy_defaults(_defaults())
    scenario = HousingFinanceScenario(
        housing_type=HousingType.BTO,
        household_profile=HouseholdProfile.SG_SG,
        purchase_price=520000,
        expected_sale_price=760000,
        years_to_sell=10,
        mop_years=5,
        wait_years_to_keys=4,
        cov_amount=0,
        renovation_cost=45000,
        monthly_other_costs=150,
        monthly_rental_income=0,
        loan_type=LoanType.HDB,
        use_hdb_loan=True,
        loan_tenure_years=25,
        downpayment_pct=0.2,
        annual_interest_rate_pct=2.6,
        cpf_oa_monthly_available=1200,
        grants=(
            GrantSelection(name="EHG", amount=60000, selected=True),
            GrantSelection(name="Family Grant", amount=50000, selected=False),
        ),
        resale_levy_amount=40000,
    )
    out = run_housing_finance(scenario, cfg)
    assert out.timeline.mop_satisfied_by_sale
    assert out.government.grants_taken_total == 60000
    assert out.costs.upfront_total > 0
    assert out.monthly_cashflow


def test_run_housing_finance_sg_pr_resale() -> None:
    cfg = load_policy_defaults(_defaults())
    scenario = HousingFinanceScenario(
        housing_type=HousingType.RESALE,
        household_profile=HouseholdProfile.SG_PR,
        purchase_price=780000,
        expected_sale_price=890000,
        years_to_sell=6,
        mop_years=5,
        wait_years_to_keys=0.5,
        cov_amount=20000,
        renovation_cost=55000,
        monthly_other_costs=180,
        monthly_rental_income=0,
        loan_type=LoanType.BANK_FIXED,
        use_hdb_loan=False,
        loan_tenure_years=30,
        downpayment_pct=0.25,
        annual_interest_rate_pct=3.3,
        cpf_oa_monthly_available=1700,
    )
    out = run_housing_finance(scenario, cfg)
    assert out.government.total_government_return >= 0
    assert out.profit.gross_sale_proceeds == 890000
    assert out.profit.loan_redemption >= 0


def test_variable_bank_rate_path() -> None:
    cfg = load_policy_defaults(_defaults())
    scenario = HousingFinanceScenario(
        housing_type=HousingType.EC,
        household_profile=HouseholdProfile.SG_SG,
        purchase_price=980000,
        expected_sale_price=1180000,
        years_to_sell=9,
        mop_years=5,
        wait_years_to_keys=3,
        loan_type=LoanType.BANK_FLOATING,
        use_hdb_loan=False,
        loan_tenure_years=25,
        downpayment_pct=0.25,
        annual_interest_rate_pct=3.0,
        cpf_oa_monthly_available=1500,
    )
    out = run_housing_finance(scenario, cfg)
    assert out.monthly_cashflow[0].annual_rate_pct >= 0
    assert out.costs.interest_total >= 0


def test_make_fixed_then_sora_segments() -> None:
    segs = make_fixed_then_sora_segments(
        tenure_years=25,
        fixed_period_months=24,
        fixed_rate_pct=3.0,
        sora_rate_pct=2.5,
        sora_spread_pct=1.0,
    )
    assert len(segs) == 2
    assert segs[0].months == 24
    assert segs[1].annual_rate_pct == 3.5


def test_bank_fixed_defaults_to_fixed_then_sora_segments() -> None:
    cfg = load_policy_defaults(_defaults())
    scenario = HousingFinanceScenario(
        housing_type=HousingType.RESALE,
        household_profile=HouseholdProfile.SG_PR,
        purchase_price=700000,
        expected_sale_price=900000,
        years_to_sell=7,
        mop_years=5,
        wait_years_to_keys=0,
        loan_type=LoanType.BANK_FIXED,
        use_hdb_loan=False,
        loan_tenure_years=25,
        downpayment_pct=0.25,
        annual_interest_rate_pct=3.1,
        cpf_oa_monthly_available=1300,
    )
    out = run_housing_finance(scenario, cfg)
    assert len(out.scenario.bank_rate_segments) >= 1
    assert out.scenario.bank_rate_segments[0].months == 24


def test_repricing_savings_computation() -> None:
    cfg = load_policy_defaults(_defaults())
    scenario = HousingFinanceScenario(
        housing_type=HousingType.RESALE,
        household_profile=HouseholdProfile.SG_SG,
        purchase_price=600000,
        expected_sale_price=820000,
        years_to_sell=8,
        mop_years=5,
        wait_years_to_keys=0.0,
        loan_type=LoanType.BANK_FIXED,
        use_hdb_loan=False,
        loan_tenure_years=25,
        downpayment_pct=0.25,
        annual_interest_rate_pct=3.6,
        cpf_oa_monthly_available=1200,
        enable_repricing=True,
        repricing_month=25,
        repricing_target_rate_pct=2.8,
        repricing_admin_fee=500,
        refinancing_legal_fee=2000,
        refinancing_valuation_fee=300,
        lock_in_months=24,
        early_repayment_penalty_pct=0.015,
        clawback_fee=0,
    )
    out = run_housing_finance(scenario, cfg)
    assert out.repricing is not None
    assert out.repricing.total_switch_cost >= 0
    assert out.repricing.baseline_interest_total >= out.repricing.repriced_interest_total
