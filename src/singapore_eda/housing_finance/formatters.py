"""Formatting helpers for housing finance outputs."""

from __future__ import annotations

import pandas as pd

from singapore_eda.housing_finance.models import HousingFinanceResult


def itemized_cost_table(result: HousingFinanceResult) -> pd.DataFrame:
    rows = [
        ("Upfront cash", result.costs.upfront_cash),
        ("Upfront CPF", result.costs.upfront_cpf),
        ("Upfront total", result.costs.upfront_total),
        ("Recurring total", result.costs.recurring_total),
        ("Interest total", result.costs.interest_total),
        ("One-off total", result.costs.one_off_total),
        ("Exit total", result.costs.exit_total),
        ("Total cost of ownership", result.costs.total_cost_of_ownership),
    ]
    return pd.DataFrame(rows, columns=["item", "amount_sgd"])


def government_return_table(result: HousingFinanceResult) -> pd.DataFrame:
    rows = [
        ("Grants taken", result.government.grants_taken_total),
        ("Estimated grant return", result.government.estimated_grant_return),
        ("Resale levy", result.government.resale_levy),
        ("Other government return", result.government.other_government_return),
        ("Total return to government", result.government.total_government_return),
    ]
    return pd.DataFrame(rows, columns=["item", "amount_sgd"])


def profit_breakdown_table(result: HousingFinanceResult) -> pd.DataFrame:
    rows = [
        ("Gross sale proceeds", result.profit.gross_sale_proceeds),
        ("Sale agent fee", result.profit.sale_agent_fee),
        ("Loan redemption", result.profit.loan_redemption),
        ("Net proceeds after obligations", result.profit.net_proceeds_after_obligations),
        ("User contributions", result.profit.user_total_contributions),
        ("Estimated profit", result.profit.estimated_profit),
    ]
    return pd.DataFrame(rows, columns=["item", "amount_sgd"])


def cashflow_table(result: HousingFinanceResult, max_rows: int = 240) -> pd.DataFrame:
    rows = result.monthly_cashflow[:max_rows]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "month": [r.month_index for r in rows],
            "opening_balance": [r.opening_balance for r in rows],
            "instalment": [r.instalment for r in rows],
            "interest": [r.interest_paid for r in rows],
            "principal": [r.principal_paid for r in rows],
            "cpf_used": [r.cpf_used for r in rows],
            "cash_used": [r.cash_used for r in rows],
            "rental_inflow": [r.rental_inflow for r in rows],
            "net_cash_outflow": [r.net_cash_outflow for r in rows],
            "annual_rate_pct": [r.annual_rate_pct for r in rows],
        }
    )
