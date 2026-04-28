#!/usr/bin/env python3
"""Build V1 forecaster artifacts and optionally load curated features to Postgres."""

from singapore_eda.forecaster_build import main  # noqa: I001


if __name__ == "__main__":
    main()
