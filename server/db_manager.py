import os
import re
import sqlite3
from typing import Any, List, Optional, Tuple


class DBManager:
    def __init__(self):
        self.backend = "sqlite"
        self.conn: Any = None
        self._shadow_ddl = ""
        self._connect()

    def _connect(self) -> None:
        force_sqlite = os.environ.get("USE_SQLITE")
        if not force_sqlite:
            try:
                import psycopg2

                self.conn = psycopg2.connect(
                    host="localhost",
                    port=int(os.environ.get("PGPORT", 5433)),
                    dbname="chronomigrate",
                    user="user",
                )
                self.conn.autocommit = False
                self.backend = "postgresql"
                return
            except Exception:
                pass

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.backend = "sqlite"

    def _recreate_sqlite(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")

    def _split_sql(self, sql: str) -> List[str]:
        return [part.strip() for part in sql.split(";") if part.strip()]

    def _normalize_sqlite_statement(self, sql: str) -> str:
        normalized = sql
        normalized = re.sub(r"\bBIGSERIAL\b", "INTEGER", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bSERIAL\b", "INTEGER", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bBOOLEAN\b", "INTEGER", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bJSONB\b", "TEXT", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bDECIMAL\(\d+,\d+\)\b", "REAL", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bNOW\(\)", "CURRENT_TIMESTAMP", normalized, flags=re.IGNORECASE)
        normalized = re.sub(
            r"\)\s*PARTITION BY\s+[A-Z]+\s*\([^)]+\)",
            ")",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        normalized = re.sub(
            r"CREATE TABLE\s+(\w+)\s+PARTITION OF\s+\w+.*",
            r"CREATE TABLE \1 (id INTEGER)",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        normalized = re.sub(
            r"ALTER TABLE\s+\w+\s+DROP CONSTRAINT\s+\w+",
            "SELECT 1",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"ALTER TABLE\s+\w+\s+ADD CONSTRAINT\s+\w+\s+FOREIGN KEY.*",
            "SELECT 1",
            normalized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return normalized

    def execute(self, sql: str, params: Optional[list] = None) -> Tuple[bool, str, list]:
        rows = []
        try:
            cursor = self.conn.cursor()
            for statement in self._split_sql(sql):
                runnable = (
                    self._normalize_sqlite_statement(statement)
                    if self.backend == "sqlite"
                    else statement
                )
                cursor.execute(runnable, params or [])
                try:
                    rows = cursor.fetchall()
                except Exception:
                    rows = []
                self._apply_shadow_schema_change(statement)
            self.conn.commit()
            return True, "SUCCESS", rows
        except Exception as exc:
            try:
                self.conn.rollback()
            except Exception:
                pass
            return False, str(exc), []

    def get_schema_ddl(self) -> str:
        return self._shadow_ddl.strip()

    def reset_to_schema(self, schema_sql: str, seed_data_sql: str) -> None:
        if self.backend == "sqlite":
            self._recreate_sqlite()
        else:
            cursor = self.conn.cursor()
            cursor.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            self.conn.commit()

        self._shadow_ddl = schema_sql.strip()
        ok, error, _ = self.execute(schema_sql)
        if not ok:
            raise RuntimeError(f"Failed to initialize schema: {error}")
        ok, error, _ = self.execute(seed_data_sql)
        if not ok:
            raise RuntimeError(f"Failed to seed data: {error}")

    def _apply_shadow_schema_change(self, sql: str) -> None:
        statement = sql.strip().rstrip(";")
        upper = statement.upper()
        if not statement:
            return

        if upper.startswith("ALTER TABLE") and "ADD COLUMN" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+ADD COLUMN\s+(.+)$",
                statement,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                table, column_def = match.groups()
                self._insert_into_table(table, column_def.strip())
            return

        if upper.startswith("ALTER TABLE") and "RENAME COLUMN" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+RENAME COLUMN\s+(\w+)\s+TO\s+(\w+)",
                statement,
                flags=re.IGNORECASE,
            )
            if match:
                table, old, new = match.groups()
                self._shadow_ddl = re.sub(
                    rf"(CREATE TABLE\s+{re.escape(table)}\s*\(.*?){re.escape(old)}(\s+)",
                    rf"\1{new}\2",
                    self._shadow_ddl,
                    count=1,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                self._shadow_ddl = re.sub(
                    rf"REFERENCES\s+{re.escape(table)}\s*\(\s*{re.escape(old)}\s*\)",
                    f"REFERENCES {table}({new})",
                    self._shadow_ddl,
                    flags=re.IGNORECASE,
                )
            return

        if upper.startswith("ALTER TABLE") and "DROP CONSTRAINT" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+DROP CONSTRAINT\s+(\w+)",
                statement,
                flags=re.IGNORECASE,
            )
            if match:
                table, constraint = match.groups()
                pattern = (
                    rf"(CREATE TABLE\s+{re.escape(table)}\s*\(.*?)(,\s*CONSTRAINT\s+"
                    rf"{re.escape(constraint)}.*?)(\)\s*;)"
                )
                self._shadow_ddl = re.sub(
                    pattern,
                    r"\1\3",
                    self._shadow_ddl,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            return

        if upper.startswith("ALTER TABLE") and "ADD CONSTRAINT" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+ADD CONSTRAINT\s+(.+)$",
                statement,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                table, constraint_def = match.groups()
                self._insert_into_table(table, f"CONSTRAINT {constraint_def.strip()}")
            return

        if upper.startswith("ALTER TABLE") and "RENAME TO" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+RENAME TO\s+(\w+)",
                statement,
                flags=re.IGNORECASE,
            )
            if match:
                old, new = match.groups()
                self._shadow_ddl = re.sub(
                    rf"\b{re.escape(old)}\b",
                    new,
                    self._shadow_ddl,
                    flags=re.IGNORECASE,
                )
            return

        if upper.startswith("CREATE TABLE"):
            if self._shadow_ddl:
                self._shadow_ddl += "\n\n" + statement + ";"
            else:
                self._shadow_ddl = statement + ";"
            return

        if upper.startswith("DROP TABLE"):
            match = re.search(r"DROP TABLE\s+(\w+)", statement, flags=re.IGNORECASE)
            if match:
                table = match.group(1)
                self._shadow_ddl = re.sub(
                    rf"CREATE TABLE\s+{re.escape(table)}.*?;\s*",
                    "",
                    self._shadow_ddl,
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip()

    def _insert_into_table(self, table: str, fragment: str) -> None:
        pattern = rf"(CREATE TABLE\s+{re.escape(table)}\s*\()(.*?)(\)\s*(?:PARTITION BY.*)?;)"
        match = re.search(pattern, self._shadow_ddl, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return

        prefix, body, suffix = match.groups()
        body = body.rstrip()
        if body and not body.strip().endswith(","):
            body += ","
        updated = f"{prefix}{body}\n    {fragment}\n{suffix}"
        self._shadow_ddl = re.sub(pattern, updated, self._shadow_ddl, count=1, flags=re.IGNORECASE | re.DOTALL)
