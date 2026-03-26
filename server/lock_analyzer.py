from dataclasses import dataclass

import sqlglot
import sqlglot.expressions as exp


@dataclass(frozen=True)
class LockProfile:
    lock_type: str
    lock_ticks: int
    failure_rate: float
    destroys_data: bool = False


def analyze_lock(sql: str) -> LockProfile:
    try:
        statement = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return LockProfile("NONE", 0, 0.0, False)

    if statement is None:
        return LockProfile("NONE", 0, 0.0, False)

    if isinstance(statement, exp.Drop) and statement.args.get("kind") == "TABLE":
        return LockProfile("ACCESS_EXCLUSIVE", 20, 1.0, True)

    if isinstance(statement, exp.TruncateTable):
        return LockProfile("ACCESS_EXCLUSIVE", 20, 1.0, True)

    if isinstance(statement, exp.Delete) and statement.args.get("where") is None:
        return LockProfile("ACCESS_EXCLUSIVE", 20, 1.0, True)

    if isinstance(statement, exp.Create) and statement.args.get("kind") == "INDEX":
        if statement.args.get("concurrently"):
            return LockProfile("SHARE_UPDATE_EXCLUSIVE", 0, 0.0, False)
        return LockProfile("SHARE", 8, 0.6, False)

    if isinstance(statement, exp.Insert) and statement.args.get("expression") is not None:
        return LockProfile("ROW_EXCLUSIVE", 5, 0.1, False)

    if isinstance(statement, exp.AlterTable):
        sql_upper = sql.upper()
        if "RENAME COLUMN" in sql_upper:
            return LockProfile("ACCESS_EXCLUSIVE", 3, 0.25, False)
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

    return LockProfile("SHARE_UPDATE_EXCLUSIVE", 1, 0.05, False)
