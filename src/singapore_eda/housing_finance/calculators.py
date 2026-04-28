"""Core calculators for housing finance details page."""

from __future__ import annotations

from dataclasses import replace
from math import isclose

from singapore_eda.housing_finance.models import (
    CostBreakdown,
    EligibilitySummary,
    GovernmentObligationSummary,
    HouseholdProfile,
    HousingFinanceResult,
    HousingFinanceScenario,
    LoanMonthRow,
    LoanType,
    ProfitSummary,
    RateSegment,
    RepricingSummary,
    TimelineSummary,
)
from singapore_eda.housing_finance.policy_defaults import PolicyDefaults


def make_fixed_then_sora_segments(
    *,
    tenure_years: int,
    fixed_period_months: int,
    fixed_rate_pct: float,
    sora_rate_pct: float,
    sora_spread_pct: float,
) -> tuple[RateSegment, ...]:
    total_months = max(1, int(tenure_years) * 12)
    fixed_months = max(0, min(int(fixed_period_months), total_months))
    floating_months = max(0, total_months - fixed_months)
    out: list[RateSegment] = []
    if fixed_months > 0:
        out.append(RateSegment(months=fixed_months, annual_rate_pct=max(0.0, fixed_rate_pct)))
    if floating_months > 0:
        out.append(
            RateSegment(
                months=floating_months,
                annual_rate_pct=max(0.0, sora_rate_pct + sora_spread_pct),
            )
        )
    return tuple(out)


def _monthly_rate(rate_pct: float) -> float:
    return rate_pct / 100.0 / 12.0


def _monthly_instalment(principal: float, annual_rate_pct: float, months: int) -> float:
    if principal <= 0 or months <= 0:
        return 0.0
    r = _monthly_rate(annual_rate_pct)
    if isclose(r, 0.0):
        return principal / float(months)
    return principal * (r * (1.0 + r) ** months) / (((1.0 + r) ** months) - 1.0)


def _rate_for_month(
    segments: tuple[RateSegment, ...],
    month_index: int,
    fallback_rate: float,
) -> float:
    if not segments:
        return fallback_rate
    remaining = month_index
    for seg in segments:
        if remaining < seg.months:
            return seg.annual_rate_pct
        remaining -= seg.months
    return segments[-1].annual_rate_pct


def apply_defaults(s: HousingFinanceScenario, defaults: PolicyDefaults) -> HousingFinanceScenario:
    grants = s.grants
    if not grants:
        grants = tuple(defaults.grants_by_household.get(s.household_profile.value, []))
    levy = s.resale_levy_amount
    if levy <= 0:
        levy = defaults.resale_levy_by_housing_type.get(s.housing_type.value, 0.0)
    legal = s.legal_fees if s.legal_fees > 0 else defaults.legal_fees_default
    valuation = s.valuation_fees if s.valuation_fees > 0 else defaults.valuation_fees_default
    bsd = s.buyer_stamp_duty
    if bsd <= 0:
        bsd = s.purchase_price * defaults.buyer_stamp_duty_default_rate_pct / 100.0
    absd = s.additional_buyer_stamp_duty
    if absd <= 0:
        absd = s.purchase_price * defaults.additional_buyer_stamp_duty_default_rate_pct / 100.0
    agent = (
        s.agent_sale_fee_pct
        if s.agent_sale_fee_pct > 0
        else defaults.agent_sale_fee_default_rate_pct
    )
    rate = s.annual_interest_rate_pct
    if s.loan_type == LoanType.HDB:
        rate = defaults.hdb_loan_rate_pct
    elif s.loan_type == LoanType.BANK_FIXED and rate <= 0:
        rate = defaults.bank_fixed_rate_pct
    elif s.loan_type == LoanType.BANK_FLOATING and rate <= 0:
        rate = defaults.bank_floating_base_rate_pct
    bank_segments = s.bank_rate_segments
    if (
        not bank_segments
        and s.loan_type == LoanType.BANK_FIXED
        and defaults.bank_fixed_then_sora_default
    ):
        bank_segments = make_fixed_then_sora_segments(
            tenure_years=s.loan_tenure_years,
            fixed_period_months=defaults.bank_fixed_period_months,
            fixed_rate_pct=max(0.0, rate if rate > 0 else defaults.bank_fixed_rate_pct),
            sora_rate_pct=defaults.bank_floating_base_rate_pct,
            sora_spread_pct=defaults.bank_sora_spread_pct,
        )
    other = (
        s.monthly_other_costs
        if s.monthly_other_costs > 0
        else defaults.maintenance_monthly_default
    )
    return replace(
        s,
        grants=tuple(grants),
        resale_levy_amount=levy,
        legal_fees=legal,
        valuation_fees=valuation,
        buyer_stamp_duty=bsd,
        additional_buyer_stamp_duty=absd,
        agent_sale_fee_pct=agent,
        annual_interest_rate_pct=rate,
        bank_rate_segments=tuple(bank_segments),
        monthly_other_costs=other,
    )


def compute_timeline(s: HousingFinanceScenario) -> TimelineSummary:
    years_to_mop = max(0.0, s.mop_years - s.wait_years_to_keys)
    earliest_sale = max(s.mop_years, s.wait_years_to_keys)
    return TimelineSummary(
        years_to_mop=years_to_mop,
        earliest_sale_year_from_purchase=earliest_sale,
        years_until_estimated_sale=s.years_to_sell,
        mop_satisfied_by_sale=s.years_to_sell >= earliest_sale,
    )


def compute_eligibility(s: HousingFinanceScenario, timeline: TimelineSummary) -> EligibilitySummary:
    messages: list[str] = []
    if not timeline.mop_satisfied_by_sale:
        messages.append("Estimated sale date occurs before MOP completion.")
    if s.household_profile == HouseholdProfile.OTHER:
        messages.append("Household profile may need manual eligibility verification.")
    if s.loan_type == LoanType.HDB and s.housing_type.value == "private":
        messages.append("HDB loan usually does not apply to private housing.")
    return EligibilitySummary(
        is_estimate_allowed=len(messages) == 0,
        messages=tuple(messages),
    )


def compute_government_obligations(
    s: HousingFinanceScenario, grant_return_rate_pct: float
) -> GovernmentObligationSummary:
    grants_taken_total = sum(g.amount for g in s.grants if g.selected)
    grant_return = grants_taken_total * (grant_return_rate_pct / 100.0)
    total = grant_return + s.resale_levy_amount + s.gov_return_extra_amount
    return GovernmentObligationSummary(
        grants_taken_total=grants_taken_total,
        estimated_grant_return=grant_return,
        resale_levy=s.resale_levy_amount,
        other_government_return=s.gov_return_extra_amount,
        total_government_return=total,
    )


def build_monthly_cashflow(s: HousingFinanceScenario) -> tuple[LoanMonthRow, ...]:
    loan_principal = s.purchase_price * (1.0 - s.downpayment_pct)
    months = s.loan_tenure_years * 12
    if months <= 0:
        return tuple()
    bal = float(loan_principal)
    rows: list[LoanMonthRow] = []
    for i in range(months):
        rate = _rate_for_month(s.bank_rate_segments, i, s.annual_interest_rate_pct)
        instal = _monthly_instalment(bal, rate, max(1, months - i))
        interest = bal * _monthly_rate(rate)
        principal = max(0.0, min(instal - interest, bal))
        end_bal = max(0.0, bal - principal)
        cpf_used = min(s.cpf_oa_monthly_available, instal)
        cash_used = max(0.0, instal - cpf_used) + s.monthly_other_costs
        rental_inflow = s.monthly_rental_income
        net_cash = cash_used - rental_inflow
        rows.append(
            LoanMonthRow(
                month_index=i + 1,
                opening_balance=bal,
                instalment=instal,
                principal_paid=principal,
                interest_paid=interest,
                cpf_used=cpf_used,
                cash_used=cash_used,
                net_cash_outflow=net_cash,
                rental_inflow=rental_inflow,
                net_inflow_outflow=(-net_cash),
                annual_rate_pct=rate,
            )
        )
        bal = end_bal
    return tuple(rows)


def compute_costs(
    s: HousingFinanceScenario,
    monthly_cashflow: tuple[LoanMonthRow, ...],
    government: GovernmentObligationSummary,
) -> CostBreakdown:
    downpayment = s.purchase_price * s.downpayment_pct
    upfront_cash = downpayment + s.cov_amount + s.renovation_cost + s.legal_fees + s.valuation_fees
    upfront_total = upfront_cash + s.buyer_stamp_duty + s.additional_buyer_stamp_duty
    recurring_total = sum(r.cash_used for r in monthly_cashflow)
    interest_total = sum(r.interest_paid for r in monthly_cashflow)
    one_off_total = s.resale_levy_amount + s.gov_return_extra_amount
    exit_total = government.total_government_return
    ownership = upfront_total + recurring_total + interest_total + one_off_total + exit_total
    return CostBreakdown(
        upfront_cash=upfront_cash,
        upfront_cpf=0.0,
        upfront_total=upfront_total,
        recurring_total=recurring_total,
        interest_total=interest_total,
        one_off_total=one_off_total,
        exit_total=exit_total,
        total_cost_of_ownership=ownership,
    )


def compute_profit(
    s: HousingFinanceScenario,
    monthly_cashflow: tuple[LoanMonthRow, ...],
    government: GovernmentObligationSummary,
    costs: CostBreakdown,
) -> ProfitSummary:
    if not monthly_cashflow:
        loan_redemption = s.purchase_price * (1.0 - s.downpayment_pct)
    else:
        loan_redemption = max(
            0.0,
            monthly_cashflow[-1].opening_balance - monthly_cashflow[-1].principal_paid,
        )
    sale_fee = s.expected_sale_price * s.agent_sale_fee_pct
    net = s.expected_sale_price - sale_fee - loan_redemption - government.total_government_return
    contributions = costs.upfront_total + sum(r.cash_used for r in monthly_cashflow)
    profit = net - contributions
    years = max(1e-9, s.years_to_sell)
    annualized = ((profit / max(1.0, contributions)) / years) * 100.0
    return ProfitSummary(
        gross_sale_proceeds=s.expected_sale_price,
        sale_agent_fee=sale_fee,
        loan_redemption=loan_redemption,
        net_proceeds_after_obligations=net,
        user_total_contributions=contributions,
        estimated_profit=profit,
        annualized_profit_rate_pct=annualized,
    )


def compute_repricing_summary(
    s: HousingFinanceScenario,
    baseline_cashflow: tuple[LoanMonthRow, ...],
) -> RepricingSummary | None:
    if not s.enable_repricing:
        return None
    if s.repricing_month <= 0 or s.repricing_target_rate_pct <= 0:
        return None
    if s.repricing_month >= len(baseline_cashflow):
        return None

    baseline_interest_total = sum(r.interest_paid for r in baseline_cashflow)
    repricing_month_idx = s.repricing_month
    opening_at_switch = baseline_cashflow[repricing_month_idx - 1].opening_balance
    remaining_months = max(1, len(baseline_cashflow) - repricing_month_idx + 1)
    switch_principal = max(0.0, opening_at_switch)

    alt_instal = _monthly_instalment(
        switch_principal,
        s.repricing_target_rate_pct,
        remaining_months,
    )
    rem_bal = switch_principal
    repriced_interest_tail = 0.0
    for _ in range(remaining_months):
        i_part = rem_bal * _monthly_rate(s.repricing_target_rate_pct)
        p_part = max(0.0, min(alt_instal - i_part, rem_bal))
        repriced_interest_tail += i_part
        rem_bal = max(0.0, rem_bal - p_part)

    baseline_interest_head = sum(
        r.interest_paid for r in baseline_cashflow[: repricing_month_idx - 1]
    )
    repriced_interest_total = baseline_interest_head + repriced_interest_tail

    penalty = 0.0
    if s.repricing_month < s.lock_in_months and s.early_repayment_penalty_pct > 0:
        penalty = opening_at_switch * s.early_repayment_penalty_pct

    fees = s.repricing_admin_fee + s.refinancing_legal_fee + s.refinancing_valuation_fee
    total_switch_cost = fees + penalty + s.clawback_fee
    gross_interest_savings = baseline_interest_total - repriced_interest_total
    net_savings = gross_interest_savings - total_switch_cost

    return RepricingSummary(
        enabled=True,
        month=s.repricing_month,
        target_rate_pct=s.repricing_target_rate_pct,
        baseline_interest_total=baseline_interest_total,
        repriced_interest_total=repriced_interest_total,
        admin_and_refinancing_fees=fees,
        early_repayment_penalty=penalty,
        clawback_fee=s.clawback_fee,
        total_switch_cost=total_switch_cost,
        gross_interest_savings=gross_interest_savings,
        net_savings=net_savings,
    )


def run_housing_finance(
    scenario: HousingFinanceScenario, defaults: PolicyDefaults
) -> HousingFinanceResult:
    s = apply_defaults(scenario, defaults)
    timeline = compute_timeline(s)
    eligibility = compute_eligibility(s, timeline)
    government = compute_government_obligations(s, defaults.grant_return_rate_pct)
    monthly_cashflow = build_monthly_cashflow(s)
    costs = compute_costs(s, monthly_cashflow, government)
    profit = compute_profit(s, monthly_cashflow, government, costs)
    repricing = compute_repricing_summary(s, monthly_cashflow)
    return HousingFinanceResult(
        scenario=s,
        timeline=timeline,
        eligibility=eligibility,
        government=government,
        costs=costs,
        profit=profit,
        monthly_cashflow=monthly_cashflow,
        repricing=repricing,
    )
