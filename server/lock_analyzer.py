from dataclasses import dataclass
import math
import re

import sqlglot
import sqlglot.expressions as exp


@dataclass(frozen=True)
class LockProfile:
    lock_type: str
    lock_ticks: int
    failure_rate: float
    destroys_data: bool = False


def _combine_profiles(left: LockProfile, right: LockProfile) -> LockProfile:
    if right.destroys_data and not left.destroys_data:
        return right
    if left.destroys_data and not right.destroys_data:
        return left
    if (right.lock_ticks, right.failure_rate) > (left.lock_ticks, left.failure_rate):
        return right
    return left


def _analyze_statement(statement: exp.Expression, sql_text: str) -> LockProfile:
    if isinstance(statement, exp.Drop) and statement.args.get("kind") == "TABLE":
        table_name = getattr(statement.this, "name", "").lower()
        if table_name.endswith("_old"):
            return LockProfile("ACCESS_EXCLUSIVE", 5, 0.25, False)
        return LockProfile("ACCESS_EXCLUSIVE", 20, 1.0, True)

    if isinstance(statement, exp.TruncateTable):
        return LockProfile("ACCESS_EXCLUSIVE", 20, 1.0, True)

    if isinstance(statement, exp.Delete) and statement.args.get("where") is None:
        return LockProfile("ACCESS_EXCLUSIVE", 20, 1.0, True)

    if isinstance(statement, exp.Create) and statement.args.get("kind") == "INDEX":
        if statement.args.get("concurrently"):
            return LockProfile("SHARE_UPDATE_EXCLUSIVE", 0, 0.0, False)
        return LockProfile("SHARE", 8, 0.6, False)

    if isinstance(statement, exp.Create) and statement.args.get("kind") == "TABLE":
        return LockProfile("ACCESS_EXCLUSIVE", 0, 0.0, False)

    if isinstance(statement, exp.Insert) and statement.args.get("expression") is not None:
        sql_upper = sql_text.upper()
        if "SELECT" in sql_upper:
            estimated_rows = 1000
            between_match = re.search(
                r"BETWEEN\s+(\d+)\s+AND\s+(\d+)",
                sql_upper,
            )
            limit_match = re.search(r"LIMIT\s+(\d+)", sql_upper)
            if between_match:
                start, end = between_match.groups()
                estimated_rows = max(1, int(end) - int(start) + 1)
            elif limit_match:
                estimated_rows = max(1, int(limit_match.group(1)))

            lock_ticks = max(1, math.ceil(estimated_rows / 1000) * 15)
            failure_rate = 0.15 if ("BETWEEN" in sql_upper or "LIMIT" in sql_upper or "WHERE" in sql_upper) else 0.6
            return LockProfile("ROW_EXCLUSIVE", lock_ticks, failure_rate, False)
        return LockProfile("ROW_EXCLUSIVE", 1, 0.02, False)

    if isinstance(statement, exp.Alter):
        sql_upper = sql_text.upper()
        if "ALTER COLUMN" in sql_upper and "TYPE" in sql_upper:
            return LockProfile("ACCESS_EXCLUSIVE", 12, 0.9, False)
        if "RENAME COLUMN" in sql_upper:
            return LockProfile("ACCESS_EXCLUSIVE", 3, 0.25, False)
        if "RENAME TO" in sql_upper:
            return LockProfile("ACCESS_EXCLUSIVE", 1, 0.05, False)
        if "DROP COLUMN" in sql_upper:
            return LockProfile("ACCESS_EXCLUSIVE", 5, 0.4, False)
        if "DROP CONSTRAINT" in sql_upper:
            return LockProfile("ACCESS_EXCLUSIVE", 2, 0.2, False)
        if "ADD CONSTRAINT" in sql_upper and "FOREIGN KEY" in sql_upper:
            return LockProfile("SHARE_ROW_EXCLUSIVE", 4, 0.3, False)
        if "ADD COLUMN" in sql_upper:
            if "DEFAULT" in sql_upper:
                return LockProfile("SHARE_UPDATE_EXCLUSIVE", 0, 0.0, False)
            return LockProfile("ACCESS_EXCLUSIVE", 10, 0.8, False)

    return LockProfile("NONE", 0, 0.0, False)


def analyze_lock(sql: str) -> LockProfile:
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return LockProfile("NONE", 0, 0.0, False)

    if not statements:
        return LockProfile("NONE", 0, 0.0, False)

    profile = LockProfile("NONE", 0, 0.0, False)
    for statement in statements:
        profile = _combine_profiles(profile, _analyze_statement(statement, statement.sql()))
    return profile
