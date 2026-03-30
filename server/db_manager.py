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

    def _expand_add_column_statements(self, statement: str) -> List[str]:
        stripped = statement.strip().rstrip(";")
        match = re.match(
            r"ALTER TABLE\s+(\w+)\s+ADD COLUMN\s+(.+)$",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return [stripped]

        table_name, remainder = match.groups()
        fragments = self._split_top_level_commas(remainder)
        if len(fragments) <= 1:
            return [stripped]

        expanded = []
        for fragment in fragments:
            normalized = re.sub(
                r"^ADD COLUMN\s+",
                "",
                fragment.strip(),
                flags=re.IGNORECASE,
            )
            expanded.append(f"ALTER TABLE {table_name} ADD COLUMN {normalized}")
        return expanded

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

    def _resolved_shadow_table_body(self, table_name: str, seen: Optional[set[str]] = None) -> str:
        seen = seen or set()
        normalized_name = table_name.lower()
        if normalized_name in seen:
            return self._shadow_table_body(table_name)
        seen.add(normalized_name)

        body = self._shadow_table_body(table_name).strip()
        like_match = re.fullmatch(
            r"LIKE\s+(\w+)\s+INCLUDING\s+ALL",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if like_match:
            return self._resolved_shadow_table_body(like_match.group(1), seen)
        return body

    def _shadow_table_names(self) -> List[str]:
        return [
            match.group(1)
            for match in re.finditer(r"CREATE TABLE\s+(\w+)\b", self._shadow_ddl, re.IGNORECASE)
        ]

    def _column_exists_in_shadow(self, table_name: str, column_name: str) -> bool:
        body = self._shadow_table_body(table_name)
        if not body:
            return False

        for clause in self._split_top_level_commas(body):
            if clause.upper().startswith("CONSTRAINT"):
                continue
            tokens = clause.strip().split()
            if tokens and tokens[0].strip('"').lower() == column_name.lower():
                return True
        return False

    def _constraint_exists_in_shadow(self, table_name: str, constraint_name: str) -> bool:
        body = self._shadow_table_body(table_name)
        if not body:
            return False
        pattern = re.compile(
            rf"\bCONSTRAINT\s+{re.escape(constraint_name)}\b",
            re.IGNORECASE,
        )
        return bool(pattern.search(body))

    def _foreign_keys_referencing(self, table_name: str, column_name: str) -> List[Tuple[str, str]]:
        references: List[Tuple[str, str]] = []
        target = rf"REFERENCES\s+{re.escape(table_name)}\s*\(\s*{re.escape(column_name)}\s*\)"
        pattern = re.compile(
            rf"\bCONSTRAINT\s+(\w+)\b.*?{target}",
            re.IGNORECASE | re.DOTALL,
        )
        for shadow_table in self._shadow_table_names():
            body = self._shadow_table_body(shadow_table)
            if not body:
                continue
            for match in pattern.finditer(body):
                references.append((shadow_table, match.group(1)))
        return references

    def _preflight_statement(self, statement: str) -> Optional[str]:
        statement = statement.strip().rstrip(";")
        upper = statement.upper()
        if not statement:
            return None

        if upper.startswith("ALTER TABLE") and "DROP CONSTRAINT" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+DROP CONSTRAINT(\s+IF EXISTS)?\s+(\w+)",
                statement,
                flags=re.IGNORECASE,
            )
            if match:
                table_name, if_exists_clause, constraint_name = match.groups()
                if if_exists_clause:
                    return None
                if not self._constraint_exists_in_shadow(table_name, constraint_name):
                    return f"constraint {constraint_name} does not exist on {table_name}"
            return None

        if upper.startswith("ALTER TABLE") and "RENAME COLUMN" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+RENAME COLUMN\s+(\w+)\s+TO\s+(\w+)",
                statement,
                flags=re.IGNORECASE,
            )
            if match:
                table_name, old_name, _ = match.groups()
                references = self._foreign_keys_referencing(table_name, old_name)
                if references:
                    details = ", ".join(
                        f"{ref_table}.{constraint_name}"
                        for ref_table, constraint_name in references
                    )
                    return (
                        f"foreign key constraint still references {table_name}({old_name}); "
                        f"drop referencing constraint(s) first: {details}"
                    )
            return None

        if upper.startswith("ALTER TABLE") and "ADD CONSTRAINT" in upper and "FOREIGN KEY" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+ADD CONSTRAINT\s+(\w+)\s+FOREIGN KEY\s*"
                r"\(\s*(\w+)\s*\)\s+REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)",
                statement,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                table_name, constraint_name, local_column, ref_table, ref_column = match.groups()
                if self._constraint_exists_in_shadow(table_name, constraint_name):
                    return f"constraint {constraint_name} already exists on {table_name}"
                if not self._column_exists_in_shadow(table_name, local_column):
                    return f"column {table_name}.{local_column} does not exist for foreign key"
                if not self._column_exists_in_shadow(ref_table, ref_column):
                    return f"referenced column {ref_table}.{ref_column} does not exist"
            return None

        return None

    def _sqlite_columns_for_table(self, table_name: str) -> str:
        body = self._resolved_shadow_table_body(table_name)
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

    def _expand_like_create_for_shadow(self, statement: str) -> str:
        stripped = statement.strip().rstrip(";")
        like_partition = re.match(
            r"CREATE TABLE\s+(\w+)\s*\(LIKE\s+(\w+)\s+INCLUDING\s+ALL\)\s*(PARTITION BY\s+[A-Z]+\s*\([^)]+\))?\s*$",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not like_partition:
            return stripped

        table_name, parent_name, partition_clause = like_partition.groups()
        parent_body = self._resolved_shadow_table_body(parent_name)
        if not parent_body:
            return stripped

        expanded = f"CREATE TABLE {table_name} (\n{parent_body.strip()}\n)"
        if partition_clause:
            expanded += f" {partition_clause.strip()}"
        return expanded

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
        if upper.startswith("DROP TABLE") and "CASCADE" in upper:
            return re.sub(r"\s+CASCADE\b", "", statement, flags=re.IGNORECASE)
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
        normalized = re.sub(
            r"\bDROP TABLE\s+(\w+)\s+CASCADE\b",
            r"DROP TABLE \1",
            normalized,
            flags=re.IGNORECASE,
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
                expanded_statements = (
                    self._expand_add_column_statements(statement)
                    if self.backend == "sqlite"
                    else [statement.strip().rstrip(";")]
                )
                for expanded_statement in expanded_statements:
                    preflight_error = self._preflight_statement(expanded_statement)
                    if preflight_error:
                        raise RuntimeError(preflight_error)
                    runnable = (
                        self._translate_sqlite_statement(expanded_statement)
                        if self.backend == "sqlite"
                        else expanded_statement
                    )
                    cursor.execute(runnable, params or [])
                    try:
                        rows = cursor.fetchall()
                    except Exception:
                        rows = []
                    if track_shadow:
                        self._apply_shadow_schema_change(expanded_statement)
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
                for fragment in self._split_top_level_commas(column_def):
                    normalized = re.sub(
                        r"^ADD COLUMN\s+",
                        "",
                        fragment.strip(),
                        flags=re.IGNORECASE,
                    )
                    self._insert_into_table(table, normalized)
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
            return

        if upper.startswith("ALTER TABLE") and "DROP CONSTRAINT" in upper:
            match = re.search(
                r"ALTER TABLE\s+(\w+)\s+DROP CONSTRAINT(?:\s+IF EXISTS)?\s+(\w+)",
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
                self._shadow_ddl = re.sub(
                    rf"\b{re.escape(old)}_p(\d+)\b",
                    rf"{new}_p\1",
                    self._shadow_ddl,
                    flags=re.IGNORECASE,
                )
            return

        if upper.startswith("CREATE TABLE"):
            statement = self._expand_like_create_for_shadow(statement)
            if self._shadow_ddl:
                self._shadow_ddl += "\n\n" + statement + ";"
            else:
                self._shadow_ddl = statement + ";"
            return

        if upper.startswith("DROP TABLE"):
            match = re.search(
                r"DROP TABLE\s+(\w+)(?:\s+CASCADE)?",
                statement,
                flags=re.IGNORECASE,
            )
            if match:
                table = match.group(1)
                cascade = "CASCADE" in upper
                child_tables = re.findall(
                    rf"CREATE TABLE\s+(\w+)\s+PARTITION OF\s+{re.escape(table)}\b",
                    self._shadow_ddl,
                    flags=re.IGNORECASE,
                )
                self._shadow_ddl = re.sub(
                    rf"CREATE TABLE\s+{re.escape(table)}.*?;\s*",
                    "",
                    self._shadow_ddl,
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip()
                if cascade:
                    for child_table in child_tables:
                        self._shadow_ddl = re.sub(
                            rf"CREATE TABLE\s+{re.escape(child_table)}.*?;\s*",
                            "",
                            self._shadow_ddl,
                            flags=re.IGNORECASE | re.DOTALL,
                        ).strip()
                self._shadow_ddl = re.sub(
                    rf"CREATE TABLE\s+\w+\s+PARTITION OF\s+{re.escape(table)}\b.*?;\s*",
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
