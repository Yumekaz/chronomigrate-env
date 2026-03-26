import hashlib
import re
import sqlite3
from typing import Any, Dict, List, Tuple

import sqlglot
import sqlglot.expressions as exp


def extract_schema_fingerprint(ddl: str) -> Dict[str, Any]:
    fingerprint: Dict[str, Any] = {
        "tables": {},
        "foreign_keys": [],
        "partitions": {},
    }

    if not ddl.strip():
        return fingerprint

    try:
        statements = sqlglot.parse(ddl, dialect="postgres")
    except Exception:
        statements = []

    for stmt in statements:
        if not isinstance(stmt, exp.Create) or stmt.args.get("kind") != "TABLE":
            continue

        table = stmt.find(exp.Table)
        if table is None:
            continue

        table_name = table.name
        columns: Dict[str, Dict[str, str]] = {}

        for col_def in stmt.find_all(exp.ColumnDef):
            col_name = col_def.name
            col_type = str(col_def.args.get("kind", "")).lower()
            default_expr = col_def.args.get("default")
            columns[col_name] = {
                "type": col_type,
                "default": str(default_expr).lower() if default_expr is not None else "",
            }

        fks: List[Tuple[str, str, str, str]] = []
        for constraint in stmt.find_all(exp.ForeignKey):
            ref = constraint.args.get("reference")
            if ref is None:
                continue
            local_cols = [c.name for c in constraint.find_all(exp.Column)]
            ref_table = ref.find(exp.Table)
            ref_cols = [c.name for c in ref.find_all(exp.Column)]
            if local_cols and ref_table and ref_cols:
                fk = (table_name, local_cols[0], ref_table.name, ref_cols[0])
                fks.append(fk)
                fingerprint["foreign_keys"].append(fk)

        partition_match = re.search(
            rf"CREATE TABLE\s+{re.escape(table_name)}\s*\(.*?\)\s*PARTITION BY\s+([A-Z]+)",
            ddl,
            re.IGNORECASE | re.DOTALL,
        )
        if partition_match:
            fingerprint["partitions"][table_name] = partition_match.group(1).upper()

        fingerprint["tables"][table_name] = {
            "columns": columns,
            "foreign_keys": fks,
        }

    return fingerprint


def compute_schema_match(current_ddl: str, target_ddl: str) -> float:
    current = extract_schema_fingerprint(current_ddl)
    target = extract_schema_fingerprint(target_ddl)

    if not target["tables"]:
        return 1.0

    total = 0.0
    score = 0.0

    for table_name, target_table in target["tables"].items():
        current_table = current["tables"].get(table_name, {"columns": {}, "foreign_keys": []})

        total += 1.0
        if table_name in current["tables"]:
            score += 1.0

        for col_name, target_col in target_table["columns"].items():
            total += 1.0
            current_col = current_table["columns"].get(col_name)
            if current_col:
                score += 0.7
                if current_col["type"] == target_col["type"]:
                    score += 0.2
                if current_col["default"] == target_col["default"]:
                    score += 0.1

        for target_fk in target_table["foreign_keys"]:
            total += 1.0
            if target_fk in current_table["foreign_keys"]:
                score += 1.0

    for table_name, partition_mode in target["partitions"].items():
        total += 1.0
        if current["partitions"].get(table_name) == partition_mode:
            score += 1.0

    return min(1.0, score / total) if total else 1.0


def _sqlite_tables(conn: Any) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [row[0] for row in cursor.fetchall()]


def _postgres_tables(conn: Any) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = %s AND table_type = 'BASE TABLE' ORDER BY table_name",
        ("public",),
    )
    return [row[0] for row in cursor.fetchall()]


def compute_data_hash(conn: Any) -> str:
    hasher = hashlib.sha256()
    module = getattr(conn.__class__, "__module__", "")
    sqlite_backend = isinstance(conn, sqlite3.Connection) or module.startswith("sqlite3")

    tables = _sqlite_tables(conn) if sqlite_backend else _postgres_tables(conn)
    cursor = conn.cursor()

    for table in tables:
        cursor.execute(f"SELECT * FROM {table} ORDER BY 1")
        rows = cursor.fetchall()
        hasher.update(table.encode("utf-8"))
        hasher.update(repr(rows).encode("utf-8"))

    return hasher.hexdigest()
