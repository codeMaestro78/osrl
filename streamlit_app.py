"""Streamlit Community Cloud entrypoint for the OS-RL-Agent dashboard.

Cloud deployments cannot read the host laptop's ``/proc`` (the whole point of
the collector) and their filesystem is ephemeral, so this entrypoint selects
between a live DB, a local DB, and the bundled recorded snapshot before the
dashboard imports its cached SQLite connection. See
``osrl/dashboard/bootstrap.py`` for the resolution order.

Run locally exactly like the Cloud app::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osrl.dashboard.bootstrap import bootstrap_database  # noqa: E402

os.environ["OSRL_DB"] = bootstrap_database(ROOT)

# Importing the module executes the dashboard script (Streamlit rerun model).
from osrl.dashboard import app  # noqa: E402,F401
