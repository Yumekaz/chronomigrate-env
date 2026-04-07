from server.tasks import TaskDefinition, coerce_grader_inputs, normalize_task_score


STARTING_SCHEMA = """
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    amount DECIMAL(10,2),
    CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_name VARCHAR(200),
    CONSTRAINT fk_items_orders FOREIGN KEY (order_id) REFERENCES orders(id)
);
"""

TARGET_SCHEMA = """
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    amount DECIMAL(10,2),
    CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_name VARCHAR(200),
    CONSTRAINT fk_items_orders FOREIGN KEY (order_id) REFERENCES orders(id)
);
"""

def _format_insert_batches(table: str, columns: str, rows: list[str], batch_size: int = 250) -> str:
    statements = []
    for index in range(0, len(rows), batch_size):
        batch = rows[index : index + batch_size]
        statements.append(
            f"INSERT INTO {table} ({columns}) VALUES\n" + ",\n".join(batch) + ";"
        )
    return "\n\n".join(statements)


def _build_seed_data() -> str:
    user_rows = [f"({index}, 'user_{index:04d}')" for index in range(1, 501)]

    order_rows = []
    for index in range(1, 1501):
        user_id = ((index - 1) % 500) + 1
        amount = 10.0 + ((index * 37) % 4000) / 100.0
        order_rows.append(f"({index}, {user_id}, {amount:.2f})")

    item_rows = []
    product_names = ["book", "pen", "bag", "mouse", "desk", "chair"]
    for index in range(1, 4501):
        order_id = ((index - 1) % 1500) + 1
        product_name = product_names[(index - 1) % len(product_names)]
        item_rows.append(f"({index}, {order_id}, '{product_name}_{index:04d}')")

    return "\n\n".join(
        [
            _format_insert_batches("users", "id, username", user_rows, batch_size=250),
            _format_insert_batches("orders", "id, user_id, amount", order_rows, batch_size=300),
            _format_insert_batches(
                "order_items", "id, order_id, product_name", item_rows, batch_size=500
            ),
        ]
    )


SEED_DATA = _build_seed_data()


def grade_medium(
    current_schema_ddl: str,
    target_schema_ddl: str,
    data_hash_before: str,
    data_hash_after: str,
    availability_pct: float,
    **_: object,
) -> float:
    from server.schema_grader import compute_schema_match

    schema_match = compute_schema_match(current_schema_ddl, target_schema_ddl)
    data_integrity = 1.0 if data_hash_before == data_hash_after else 0.0
    return normalize_task_score(schema_match * data_integrity * availability_pct)


def medium_grader(*args: object, **kwargs: object) -> float:
    payload, is_complete = coerce_grader_inputs(*args, **kwargs)
    if not is_complete:
        return normalize_task_score(0.0)
    return grade_medium(**payload)


class MediumGrader:
    def __new__(cls, *args: object, **kwargs: object):
        if args or kwargs:
            return medium_grader(*args, **kwargs)
        return super().__new__(cls)

    @staticmethod
    def grade(*args: object, **kwargs: object) -> float:
        return medium_grader(*args, **kwargs)

    def __call__(self, *args: object, **kwargs: object) -> float:
        return self.grade(*args, **kwargs)


TASK = TaskDefinition(
    task_id="medium_rename_fk",
    description="Rename a primary-key column and repair foreign key references safely.",
    difficulty="medium",
    load_level=300,
    max_steps=10,
    starting_schema_sql=STARTING_SCHEMA.strip(),
    target_schema_ddl=TARGET_SCHEMA.strip(),
    seed_data_sql=SEED_DATA.strip(),
    grade_fn=grade_medium,
)
