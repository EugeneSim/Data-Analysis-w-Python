from __future__ import annotations

import pytest

from singapore_eda.feature_store import validate_identifier


def test_validate_identifier_accepts_safe_names() -> None:
    assert validate_identifier("public", label="schema_name") == "public"
    assert validate_identifier("hdb_forecaster_1", label="table_name") == "hdb_forecaster_1"


@pytest.mark.parametrize("bad", ["", "public.users", "foo-bar", "drop table x", "123abc"])
def test_validate_identifier_rejects_malformed_values(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_identifier(bad, label="schema_name")
