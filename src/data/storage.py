"""
data/storage.py
---------------
Lightweight SQLite-backed persistence layer for daily risk snapshots.

Each snapshot is a Python dict serialised to JSON and stored alongside its
date key.  This gives us a simple audit trail without requiring a full
database server.
"""

import json
import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    date        TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_connection(db_path: str) -> sqlite3.Connection:
    """
    Open (or create) a SQLite database at *db_path* and ensure the
    ``snapshots`` table exists.

    Parameters
    ----------
    db_path : str
        File-system path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        An open connection with ``check_same_thread=False`` so the same
        connection object can safely be used from a single thread.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def save_snapshot(data: dict, date: str, db_path: str = "snapshots.db") -> None:
    """
    Persist *data* as a JSON string keyed by *date*.

    If a snapshot already exists for *date* it is overwritten (UPSERT
    semantics) so re-running the pipeline on the same day is idempotent.

    Parameters
    ----------
    data : dict
        Arbitrary serialisable dictionary (portfolio metrics, Greeks, etc.).
    date : str
        ISO-8601 date string used as the primary key, e.g. ``"2025-01-15"``.
    db_path : str, optional
        Path to the SQLite file.  Defaults to ``"snapshots.db"`` in the
        current working directory.

    Raises
    ------
    Exception
        Propagates any SQLite or JSON serialisation errors after logging them.
    """
    try:
        payload_json = json.dumps(data, default=str)
        conn = _get_connection(db_path)
        conn.execute(
            """
            INSERT INTO snapshots (date, payload)
            VALUES (?, ?)
            ON CONFLICT(date) DO UPDATE SET
                payload    = excluded.payload,
                created_at = CURRENT_TIMESTAMP
            """,
            (date, payload_json),
        )
        conn.commit()
        conn.close()
        logger.info("Snapshot saved for date=%s in %s", date, db_path)
    except Exception as exc:
        logger.error("Failed to save snapshot for date=%s: %s", date, exc)
        raise


def load_snapshot(date: str, db_path: str = "snapshots.db") -> Optional[dict]:
    """
    Retrieve the snapshot stored under *date*.

    Parameters
    ----------
    date : str
        ISO-8601 date string, e.g. ``"2025-01-15"``.
    db_path : str, optional
        Path to the SQLite file.  Defaults to ``"snapshots.db"``.

    Returns
    -------
    dict or None
        The deserialised snapshot dictionary, or ``None`` if no record exists
        for *date* or if the database file does not yet exist.
    """
    try:
        conn = _get_connection(db_path)
        cursor = conn.execute(
            "SELECT payload FROM snapshots WHERE date = ?", (date,)
        )
        row = cursor.fetchone()
        conn.close()
        if row is None:
            logger.info("No snapshot found for date=%s in %s", date, db_path)
            return None
        snapshot = json.loads(row[0])
        logger.info("Snapshot loaded for date=%s", date)
        return snapshot
    except Exception as exc:
        logger.warning("Could not load snapshot for date=%s: %s", date, exc)
        return None
