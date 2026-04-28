"""Typed models for housing finance scenario calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class HousingType(StrEnum):
    BTO = "bto"
    RESALE = "resale"
    EC = "ec"
    PRIVATE = "private"


class HouseholdProfile(StrEnum):
    SG_SG = "sg_sg"
    SG_PR = "sg_pr"
    PR_PR = "pr_pr"
    SINGLE_CITIZEN = "single_citizen"
    OTHER = "other"


class LoanType(StrEnum):
    HDB = "hdb"
    BANK_FIXED = "bank_fixed"
    BANK_FLOATING = "bank_floating"


@dataclass(frozen=True)
class GrantSelection:
    name: str
    amount: float
    selected: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("GrantSelection.name must be non-empty")
        if self.amount < 0:
            raise ValueError("GrantSelection.amount must be >= 0")


@dataclass(frozen=True)
class RateSegment:
    months: int
    annual_rate_pct: float

    def __post_init__(self) -> None:
        if self.months <= 0:
            raise ValueError("RateSegment.months must be > 0")
        if self.annual_rate_pct < 0:
            raise ValueError("RateSegment.annual_rate_pct must be >= 0")


@dataclass(frozen=True)
class HousingFinanceScenario:
    housing_type: HousingType
    household_profile: HouseholdProfile
    purchase_price: float
    expected_sale_price: float
    years_to_sell: float
    mop_years: float
    wait_years_to_keys: float
    cov_amount: float = 0.0
    renovation_cost: float = 0.0
    monthly_other_costs: float = 0.0
    monthly_rental_income: float = 0.0
    annual_income: float = 0.0
    use_hdb_loan: bool = True
    loan_type: LoanType = LoanType.HDB
    loan_tenure_years: int = 25
    downpayment_pct: float = 0.2
    annual_interest_rate_pct: float = 2.6
    bank_rate_segments: tuple[RateSegment, ...] = field(default_factory=tuple)
    cpf_oa_monthly_available: float = 0.0
    grants: tuple[GrantSelection, ...] = field(default_factory=tuple)
    resale_levy_amount: float = 0.0
    gov_return_extra_amount: float = 0.0
    legal_fees: float = 0.0
    valuation_fees: float = 0.0
    buyer_stamp_duty: float = 0.0
    additional_buyer_stamp_duty: float = 0.0
    agent_sale_fee_pct: float = 0.0
    enable_repricing: bool = False
    repricing_month: int = 0
    repricing_target_rate_pct: float = 0.0
    repricing_admin_fee: float = 0.0
    refinancing_legal_fee: float = 0.0
    refinancing_valuation_fee: float = 0.0
    lock_in_months: int = 24
    early_repayment_penalty_pct: float = 0.0
    clawback_fee: float = 0.0

    def __post_init__(self) -> None:
        non_negative_fields = {
            "purchase_price": self.purchase_price,
            "expected_sale_price": self.expected_sale_price,
            "years_to_sell": self.years_to_sell,
            "mop_years": self.mop_years,
            "wait_years_to_keys": self.wait_years_to_keys,
            "cov_amount": self.cov_amount,
            "renovation_cost": self.renovation_cost,
            "monthly_other_costs": self.monthly_other_costs,
            "monthly_rental_income": self.monthly_rental_income,
            "annual_income": self.annual_income,
            "downpayment_pct": self.downpayment_pct,
            "annual_interest_rate_pct": self.annual_interest_rate_pct,
            "cpf_oa_monthly_available": self.cpf_oa_monthly_available,
            "resale_levy_amount": self.resale_levy_amount,
            "gov_return_extra_amount": self.gov_return_extra_amount,
            "legal_fees": self.legal_fees,
            "valuation_fees": self.valuation_fees,
            "buyer_stamp_duty": self.buyer_stamp_duty,
            "additional_buyer_stamp_duty": self.additional_buyer_stamp_duty,
            "agent_sale_fee_pct": self.agent_sale_fee_pct,
            "repricing_target_rate_pct": self.repricing_target_rate_pct,
            "repricing_admin_fee": self.repricing_admin_fee,
            "refinancing_legal_fee": self.refinancing_legal_fee,
            "refinancing_valuation_fee": self.refinancing_valuation_fee,
            "early_repayment_penalty_pct": self.early_repayment_penalty_pct,
            "clawback_fee": self.clawback_fee,
        }
        for k, v in non_negative_fields.items():
            if v < 0:
                raise ValueError(f"HousingFinanceScenario.{k} must be >= 0")
        if self.purchase_price <= 0:
            raise ValueError("purchase_price must be > 0")
        if self.loan_tenure_years <= 0:
            raise ValueError("loan_tenure_years must be > 0")
        if self.repricing_month < 0:
            raise ValueError("repricing_month must be >= 0")
        if self.lock_in_months < 0:
            raise ValueError("lock_in_months must be >= 0")
        if self.downpayment_pct >= 1:
            raise ValueError("downpayment_pct must be < 1")
        if self.agent_sale_fee_pct >= 1:
            raise ValueError("agent_sale_fee_pct must be < 1")
        if self.early_repayment_penalty_pct >= 1:
            raise ValueError("early_repayment_penalty_pct must be < 1")
        if self.use_hdb_loan and self.loan_type != LoanType.HDB:
            raise ValueError("loan_type must be HDB when use_hdb_loan is True")


@dataclass(frozen=True)
class LoanMonthRow:
    month_index: int
    opening_balance: float
    instalment: float
    principal_paid: float
    interest_paid: float
    cpf_used: float
    cash_used: float
    net_cash_outflow: float
    rental_inflow: float
    net_inflow_outflow: float
    annual_rate_pct: float


@dataclass(frozen=True)
class TimelineSummary:
    years_to_mop: float
    earliest_sale_year_from_purchase: float
    years_until_estimated_sale: float
    mop_satisfied_by_sale: bool


@dataclass(frozen=True)
class EligibilitySummary:
    is_estimate_allowed: bool
    messages: tuple[str, ...]


@dataclass(frozen=True)
class GovernmentObligationSummary:
    grants_taken_total: float
    estimated_grant_return: float
    resale_levy: float
    other_government_return: float
    total_government_return: float


@dataclass(frozen=True)
class CostBreakdown:
    upfront_cash: float
    upfront_cpf: float
    upfront_total: float
    recurring_total: float
    interest_total: float
    one_off_total: float
    exit_total: float
    total_cost_of_ownership: float


@dataclass(frozen=True)
class ProfitSummary:
    gross_sale_proceeds: float
    sale_agent_fee: float
    loan_redemption: float
    net_proceeds_after_obligations: float
    user_total_contributions: float
    estimated_profit: float
    annualized_profit_rate_pct: float


@dataclass(frozen=True)
class RepricingSummary:
    enabled: bool
    month: int
    target_rate_pct: float
    baseline_interest_total: float
    repriced_interest_total: float
    admin_and_refinancing_fees: float
    early_repayment_penalty: float
    clawback_fee: float
    total_switch_cost: float
    gross_interest_savings: float
    net_savings: float


@dataclass(frozen=True)
class HousingFinanceResult:
    scenario: HousingFinanceScenario
    timeline: TimelineSummary
    eligibility: EligibilitySummary
    government: GovernmentObligationSummary
    costs: CostBreakdown
    profit: ProfitSummary
    monthly_cashflow: tuple[LoanMonthRow, ...]
    repricing: RepricingSummary | None = None
