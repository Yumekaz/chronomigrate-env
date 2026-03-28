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
        self.conn.execute("PRAGMA recursive_triggers = ON")

    def _split_sql(self, sql: str) -> List[str]:
        return [part.strip() for part in sql.split(";") if part.strip()]

    def _split_top_level_commas(self, text: str) -> List[str]:
        parts: List[str] = []
        depth = 0
        start = 0
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                parts.append(text[start:index].strip())
                start = index + 1
        tail = text[start:].strip()
        if tail:
            parts.append(tail)
        return parts

    def _extract_table_body(self, ddl: str, table_name: str) -> str:
        pattern = re.compile(rf"CREATE TABLE\s+{re.escape(table_name)}\b", re.IGNORECASE)
        match = pattern.search(ddl)
        if not match:
            return ""

        start = ddl.find("(", match.end())
        if start == -1:
            return ""

        depth = 0
        for index in range(start, len(ddl)):
            char = ddl[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return ddl[start + 1 : index]
        return ""

    def _shadow_table_body(self, table_name: str) -> str:
        return self._extract_table_body(self._shadow_ddl, table_name)

    def _sqlite_columns_for_table(self, table_name: str) -> str:
        body = self._shadow_table_body(table_name)
        if not body:
            return "id INTEGER"

        clauses = []
        for clause in self._split_top_level_commas(body):
            if clause.upper().startswith("CONSTRAINT"):
                continue
            clauses.append(clause)

        if not clauses:
            return "id INTEGER"

        translated = []
        for clause in clauses:
            translated.append(self._normalize_sqlite_statement(clause))
        return ", ".join(translated)

    def _translate_create_table_sqlite(self, statement: str) -> str:
        stripped = statement.strip().rstrip(";")

        like_partition = re.match(
            r"CREATE TABLE\s+(\w+)\s*\(LIKE\s+(\w+)\s+INCLUDING\s+ALL\)\s*(?:PARTITION BY\s+[A-Z]+\s*\([^)]+\))?\s*$",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if like_partition:
            table_name, parent_name = like_partition.groups()
            columns = self._sqlite_columns_for_table(parent_name)
            return f"CREATE TABLE {table_name} ({columns})"

        partition_child = re.match(
            r"CREATE TABLE\s+(\w+)\s+PARTITION OF\s+(\w+)\s+FOR VALUES.*$",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if partition_child:
            table_name, parent_name = partition_child.groups()
            columns = self._sqlite_columns_for_table(parent_name)
            return f"CREATE TABLE {table_name} ({columns})"

        normalized = re.sub(
            r"\)\s*PARTITION BY\s+[A-Z]+\s*\([^)]+\)\s*$",
            ")",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return self._normalize_sqlite_statement(normalized)

    def _translate_sqlite_statement(self, statement: str) -> str:
        upper = statement.strip().upper()
        if upper.startswith("CREATE TABLE"):
            return self._translate_create_table_sqlite(statement)
        if upper.startswith("ALTER TABLE") and "DROP CONSTRAINT" in upper:
            return "SELECT 1"
        if upper.startswith("ALTER TABLE") and "ADD CONSTRAINT" in upper:
            return "SELECT 1"
        return self._normalize_sqlite_statement(statement)

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

    def execute(
        self,
        sql: str,
        params: Optional[list] = None,
        execute_mode: str = "transaction",
        track_shadow: bool = True,
    ) -> Tuple[bool, str, list]:
        rows = []
        execute_mode = execute_mode if execute_mode in {"transaction", "autocommit"} else "transaction"
        postgres_autocommit_prev = None
        sqlite_isolation_prev = None
        shadow_snapshot = self._shadow_ddl if track_shadow and execute_mode == "transaction" else None
        try:
            if self.backend == "postgresql" and execute_mode == "autocommit":
                postgres_autocommit_prev = self.conn.autocommit
                self.conn.autocommit = True
            if self.backend == "sqlite" and execute_mode == "autocommit":
                sqlite_isolation_prev = self.conn.isolation_level
                self.conn.isolation_level = None

            cursor = self.conn.cursor()
            for statement in self._split_sql(sql):
                runnable = self._translate_sqlite_statement(statement) if self.backend == "sqlite" else statement
                cursor.execute(runnable, params or [])
                try:
                    rows = cursor.fetchall()
                except Exception:
                    rows = []
                if track_shadow:
                    self._apply_shadow_schema_change(statement)
                if execute_mode == "autocommit" and self.backend == "sqlite":
                    self.conn.commit()
            if execute_mode == "transaction":
                self.conn.commit()
            return True, "SUCCESS", rows
        except Exception as exc:
            try:
                if execute_mode == "transaction":
                    self.conn.rollback()
                    if shadow_snapshot is not None:
                        self._shadow_ddl = shadow_snapshot
            except Exception:
                pass
            return False, str(exc), []
        finally:
            if self.backend == "postgresql" and postgres_autocommit_prev is not None:
                self.conn.autocommit = postgres_autocommit_prev
            if self.backend == "sqlite" and sqlite_isolation_prev is not None:
                self.conn.isolation_level = sqlite_isolation_prev

    def get_schema_ddl(self) -> str:
        return self._shadow_ddl.strip()

    def reset_to_schema(self, schema_sql: str, seed_data_sql: str) -> None:
        if self.backend == "sqlite":
            self._recreate_sqlite()
        else:
            cursor = self.conn.cursor()
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            self.conn.commit()

        self._shadow_ddl = ""
        ok, error, _ = self.execute(schema_sql, track_shadow=True)
        if not ok:
            raise RuntimeError(f"Failed to initialize schema: {error}")
        ok, error, _ = self.execute(seed_data_sql, track_shadow=False)
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
