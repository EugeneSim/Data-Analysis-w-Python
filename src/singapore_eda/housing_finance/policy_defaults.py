"""Policy defaults loader for housing finance calculator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from singapore_eda.housing_finance.models import (
    GrantSelection,
    HouseholdProfile,
    HousingType,
)


@dataclass(frozen=True)
class PolicyDefaults:
    effective_date: str
    updated_on: str
    disclaimer: str
    hdb_loan_rate_pct: float
    bank_fixed_rate_pct: float
    bank_floating_base_rate_pct: float
    bank_fixed_then_sora_default: bool
    bank_fixed_period_months: int
    bank_sora_spread_pct: float
    grant_return_rate_pct: float
    grants_by_household: dict[str, list[GrantSelection]]
    resale_levy_by_housing_type: dict[str, float]
    legal_fees_default: float
    valuation_fees_default: float
    buyer_stamp_duty_default_rate_pct: float
    additional_buyer_stamp_duty_default_rate_pct: float
    agent_sale_fee_default_rate_pct: float
    maintenance_monthly_default: float


def _to_grants(raw: list[dict[str, Any]]) -> list[GrantSelection]:
    out: list[GrantSelection] = []
    for row in raw:
        out.append(
            GrantSelection(
                name=str(row.get("name", "")).strip(),
                amount=float(row.get("amount", 0.0)),
                selected=bool(row.get("selected", True)),
            )
        )
    return out


def load_policy_defaults(path: Path | str) -> PolicyDefaults:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    meta = raw.get("metadata", {})
    loans = raw.get("loan_defaults", {})
    grants = raw.get("grant_defaults", {})
    fees = raw.get("fee_defaults", {})

    grants_by_household: dict[str, list[GrantSelection]] = {}
    for profile in HouseholdProfile:
        rows = grants.get(profile.value, [])
        grants_by_household[profile.value] = _to_grants(rows)

    resale_levy_by_housing_type: dict[str, float] = {}
    levy_raw = raw.get("resale_levy_defaults", {})
    for htype in HousingType:
        resale_levy_by_housing_type[htype.value] = float(levy_raw.get(htype.value, 0.0))

    return PolicyDefaults(
        effective_date=str(meta.get("effective_date", "unknown")),
        updated_on=str(meta.get("updated_on", "unknown")),
        disclaimer=str(meta.get("disclaimer", "")),
        hdb_loan_rate_pct=float(loans.get("hdb_fixed_rate_pct", 2.6)),
        bank_fixed_rate_pct=float(loans.get("bank_fixed_rate_pct", 3.2)),
        bank_floating_base_rate_pct=float(loans.get("bank_floating_base_rate_pct", 3.0)),
        bank_fixed_then_sora_default=bool(loans.get("bank_fixed_then_sora_default", True)),
        bank_fixed_period_months=int(loans.get("bank_fixed_period_months", 24)),
        bank_sora_spread_pct=float(loans.get("bank_sora_spread_pct", 1.0)),
        grant_return_rate_pct=float(grants.get("grant_return_rate_pct", 0.0)),
        grants_by_household=grants_by_household,
        resale_levy_by_housing_type=resale_levy_by_housing_type,
        legal_fees_default=float(fees.get("legal_fees_default", 2500.0)),
        valuation_fees_default=float(fees.get("valuation_fees_default", 500.0)),
        buyer_stamp_duty_default_rate_pct=float(
            fees.get("buyer_stamp_duty_default_rate_pct", 2.5)
        ),
        additional_buyer_stamp_duty_default_rate_pct=float(
            fees.get("additional_buyer_stamp_duty_default_rate_pct", 0.0)
        ),
        agent_sale_fee_default_rate_pct=float(fees.get("agent_sale_fee_default_rate_pct", 0.02)),
        maintenance_monthly_default=float(fees.get("maintenance_monthly_default", 100.0)),
    )
