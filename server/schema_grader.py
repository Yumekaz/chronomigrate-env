import hashlib
import re
import sqlite3
from typing import Any, Dict, List, Set, Tuple

import sqlglot
import sqlglot.expressions as exp


def _normalize_partition_child_name(name: str) -> str:
    match = re.search(r"_p(\d+)$", name.lower())
    return f"p{match.group(1)}" if match else name.lower()


def extract_schema_fingerprint(ddl: str) -> Dict[str, Any]:
    fingerprint: Dict[str, Any] = {
        "tables": {},
        "foreign_keys": [],
        "partitions": {},
        "partition_children": {},
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
        primary_keys: Set[str] = set()

        for col_def in stmt.find_all(exp.ColumnDef):
            col_name = col_def.name
            col_type = str(col_def.args.get("kind", "")).lower()
            default_expr = col_def.args.get("default")
            constraints = list(col_def.find_all(exp.ColumnConstraint))
            not_null = any(
                getattr(constraint.args.get("kind"), "key", "").upper() == "NOTNULL"
                for constraint in constraints
            )
            is_primary = any(
                getattr(constraint.args.get("kind"), "key", "").upper() == "PRIMARYKEY"
                for constraint in constraints
            )
            if is_primary:
                primary_keys.add(col_name)
            columns[col_name] = {
                "type": col_type,
                "default": str(default_expr).lower() if default_expr is not None else "",
                "not_null": str(not_null).lower(),
                "primary": str(is_primary).lower(),
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

        child_matches = re.findall(
            rf"CREATE TABLE\s+(\w+)\s+PARTITION OF\s+{re.escape(table_name)}\b",
            ddl,
            flags=re.IGNORECASE,
        )
        fingerprint["partition_children"][table_name] = set(child_matches)

        fingerprint["tables"][table_name] = {
            "columns": columns,
            "foreign_keys": fks,
            "primary_keys": sorted(primary_keys),
        }

    return fingerprint


def compute_schema_match(current_ddl: str, target_ddl: str) -> float:
    current = extract_schema_fingerprint(current_ddl)
    target = extract_schema_fingerprint(target_ddl)

    if not target["tables"]:
        return 1.0

    target_partition_tables = {
        child
        for children in target["partition_children"].values()
        for child in children
    }
    total = 0.0
    score = 0.0

    for table_name, target_table in target["tables"].items():
        if table_name in target_partition_tables:
            continue
        current_table = current["tables"].get(table_name, {"columns": {}, "foreign_keys": []})

        total += 1.0
        if table_name in current["tables"]:
            score += 1.0

        for col_name, target_col in target_table["columns"].items():
            total += 1.0
            current_col = current_table["columns"].get(col_name)
            if current_col:
                score += 0.5
                if current_col["type"] == target_col["type"]:
                    score += 0.2
                if current_col["default"] == target_col["default"]:
                    score += 0.1
                if current_col["not_null"] == target_col["not_null"]:
                    score += 0.1
                if current_col["primary"] == target_col["primary"]:
                    score += 0.1

        for target_fk in target_table["foreign_keys"]:
            total += 1.0
            if target_fk in current_table["foreign_keys"]:
                score += 1.0

    for table_name, partition_mode in target["partitions"].items():
        total += 1.0
        if current["partitions"].get(table_name) == partition_mode:
            score += 1.0

        target_children = {
            _normalize_partition_child_name(child)
            for child in target["partition_children"].get(table_name, set())
        }
        current_children = {
            _normalize_partition_child_name(child)
            for child in current["partition_children"].get(table_name, set())
        }
        for child in target_children:
            total += 0.5
            if child in current_children:
                score += 0.5

    base_score = min(1.0, score / total) if total else 1.0

    extra_columns = 0
    extra_foreign_keys = 0
    extra_partition_children = 0
    extra_partition_modes = 0

    for table_name, current_table in current["tables"].items():
        target_table = target["tables"].get(table_name)
        if target_table is None:
            continue

        extra_columns += len(set(current_table["columns"]) - set(target_table["columns"]))
        extra_foreign_keys += len(
            set(current_table["foreign_keys"]) - set(target_table["foreign_keys"])
        )

    for table_name, current_children in current["partition_children"].items():
        target_children = target["partition_children"].get(table_name, set())
        normalized_current_children = {
            _normalize_partition_child_name(child) for child in current_children
        }
        normalized_target_children = {
            _normalize_partition_child_name(child) for child in target_children
        }
        extra_partition_children += len(
            normalized_current_children - normalized_target_children
        )

    for table_name, current_mode in current["partitions"].items():
        target_mode = target["partitions"].get(table_name)
        if target_mode is None:
            continue
        if current_mode != target_mode:
            extra_partition_modes += 1

    penalty = min(
        0.2,
        0.03 * extra_columns
        + 0.04 * extra_foreign_keys
        + 0.02 * extra_partition_children
        + 0.05 * extra_partition_modes,
    )
    return max(0.0, min(1.0, base_score - penalty))


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


def _hash_plan_from_schema(schema_ddl: str) -> Dict[str, int]:
    fingerprint = extract_schema_fingerprint(schema_ddl)
    partition_children = {
        match.group(1)
        for match in re.finditer(
            r"CREATE TABLE\s+(\w+)\s+PARTITION OF\s+\w+",
            schema_ddl,
            re.IGNORECASE,
        )
    }
    return {
        table_name: len(table_data["columns"])
        for table_name, table_data in fingerprint["tables"].items()
        if table_name not in partition_children
    }


def compute_data_hash(conn: Any, schema_ddl: str | None = None) -> str:
    hasher = hashlib.sha256()
    module = getattr(conn.__class__, "__module__", "")
    sqlite_backend = isinstance(conn, sqlite3.Connection) or module.startswith("sqlite3")

    plan = _hash_plan_from_schema(schema_ddl) if schema_ddl else {}
    tables = sorted(plan) if plan else (_sqlite_tables(conn) if sqlite_backend else _postgres_tables(conn))
    cursor = conn.cursor()

    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table} ORDER BY 1")
            rows = cursor.fetchall()
        except Exception:
            if not sqlite_backend:
                try:
                    conn.rollback()
                except Exception:
                    pass
                cursor = conn.cursor()
            hasher.update(table.encode("utf-8"))
            hasher.update(b"__MISSING__")
            continue

        expected_columns = plan.get(table)
        if expected_columns is not None:
            rows = [tuple(row[:expected_columns]) for row in rows]
        hasher.update(table.encode("utf-8"))
        hasher.update(repr(rows).encode("utf-8"))

    return hasher.hexdigest()
