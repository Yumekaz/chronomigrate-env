import hashlib
import re
import sqlite3
from typing import Any, Dict, List, Set, Tuple

import sqlglot
import sqlglot.expressions as exp


def _normalize_partition_child_name(name: str) -> str:
    match = re.search(r"_p(\d+)$", name.lower())
    return f"p{match.group(1)}" if match else name.lower()


def _safe_sql(expression: Any) -> str:
    if expression is None:
        return ""
    try:
        return expression.sql(dialect="postgres")
    except Exception:
        return str(expression)


def _create_table_name(statement: exp.Create) -> str:
    relation = statement.args.get("this")
    if isinstance(relation, exp.Schema):
        relation = relation.this
    if isinstance(relation, exp.Table):
        return relation.name
    table = statement.find(exp.Table)
    return table.name if table is not None else ""


def _column_constraint_flags(column_def: exp.ColumnDef) -> tuple[bool, bool]:
    constraints = list(column_def.args.get("constraints") or [])
    not_null = any(
        getattr(constraint.args.get("kind"), "key", "").upper() == "NOTNULL"
        for constraint in constraints
    )
    is_primary = any(
        getattr(constraint.args.get("kind"), "key", "").upper() == "PRIMARYKEY"
        for constraint in constraints
    )
    return not_null, is_primary


def _partition_by_signature(properties: Any) -> Dict[str, Any]:
    if not isinstance(properties, exp.Properties):
        return {}

    for prop in properties.expressions:
        if isinstance(prop, exp.PartitionedByProperty):
            partition_sql = _safe_sql(prop.this).strip()
            return {
                "mode": partition_sql.split("(", 1)[0].strip().upper(),
                "columns": tuple(column.name for column in prop.this.find_all(exp.Column)),
            }
    return {}


def _partition_child_signature(properties: Any) -> Dict[str, Any]:
    if not isinstance(properties, exp.Properties):
        return {}

    for prop in properties.expressions:
        if isinstance(prop, exp.PartitionedOfProperty):
            parent = prop.this.name if isinstance(prop.this, exp.Table) else _safe_sql(prop.this)
            return {
                "parent": parent,
                "bounds": _safe_sql(prop.expression).strip().lower(),
            }
    return {}


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

        table_name = _create_table_name(stmt)
        if not table_name:
            continue

        relation = stmt.args.get("this")
        schema = relation if isinstance(relation, exp.Schema) else None
        columns: Dict[str, Dict[str, str]] = {}
        primary_keys: Set[str] = set()

        column_defs: List[exp.ColumnDef] = []
        if schema is not None:
            column_defs = [expr for expr in schema.expressions if isinstance(expr, exp.ColumnDef)]
        else:
            column_defs = list(stmt.find_all(exp.ColumnDef))

        for col_def in column_defs:
            col_name = col_def.name
            col_type = _safe_sql(col_def.args.get("kind")).lower()
            default_expr = col_def.args.get("default")
            default_value = _safe_sql(default_expr).lower() if default_expr is not None else ""
            not_null, is_primary = _column_constraint_flags(col_def)
            if is_primary:
                primary_keys.add(col_name)
            columns[col_name] = {
                "type": col_type,
                "default": default_value,
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

        properties = stmt.args.get("properties")
        partition_by = _partition_by_signature(properties)
        if partition_by:
            fingerprint["partitions"][table_name] = partition_by

        partition_child = _partition_child_signature(properties)
        if partition_child:
            parent_name = partition_child["parent"]
            fingerprint["partition_children"].setdefault(parent_name, {})[table_name] = partition_child[
                "bounds"
            ]

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

    for table_name, partition_data in target["partitions"].items():
        total += 1.0
        current_partition = current["partitions"].get(table_name)
        if current_partition:
            if current_partition.get("mode") == partition_data.get("mode"):
                score += 0.6
            if current_partition.get("columns") == partition_data.get("columns"):
                score += 0.4

        target_children = target["partition_children"].get(table_name, {})
        current_children = current["partition_children"].get(table_name, {})
        normalized_current_children = {
            _normalize_partition_child_name(child): bounds
            for child, bounds in current_children.items()
        }
        for child, target_bounds in target_children.items():
            total += 0.5
            normalized_child = _normalize_partition_child_name(child)
            current_bounds = current_children.get(child)
            if current_bounds == target_bounds:
                score += 0.5
            elif normalized_child in normalized_current_children:
                score += 0.25

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
        target_children = target["partition_children"].get(table_name, {})
        normalized_current_children = {
            _normalize_partition_child_name(child) for child in current_children
        }
        normalized_target_children = {
            _normalize_partition_child_name(child) for child in target_children
        }
        extra_partition_children += len(
            normalized_current_children - normalized_target_children
        )

    for table_name, current_partition in current["partitions"].items():
        target_partition = target["partitions"].get(table_name)
        if target_partition is None:
            continue
        if current_partition != target_partition:
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
