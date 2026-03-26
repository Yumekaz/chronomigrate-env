from server.schema_grader import compute_schema_match
from server.tasks import TaskDefinition


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

SEED_DATA = """
INSERT INTO users (id, username) VALUES
(1, 'alice'),
(2, 'bob'),
(3, 'carol');

INSERT INTO orders (id, user_id, amount) VALUES
(1, 1, 19.99),
(2, 1, 29.99),
(3, 2, 39.99);

INSERT INTO order_items (id, order_id, product_name) VALUES
(1, 1, 'book'),
(2, 1, 'pen'),
(3, 2, 'bag');
"""


def grade_medium(
    current_schema_ddl: str,
    target_schema_ddl: str,
    data_hash_before: str,
    data_hash_after: str,
    availability_pct: float,
) -> float:
    schema_match = compute_schema_match(current_schema_ddl, target_schema_ddl)
    data_integrity = 1.0 if data_hash_before == data_hash_after else 0.0
    partial_bonus = 0.1 if schema_match >= 0.7 else 0.0
    return min(1.0, schema_match * data_integrity * availability_pct + partial_bonus)


TASK = TaskDefinition(
    task_id="medium_rename_fk",
    description="Rename a primary-key column and repair foreign key references safely.",
    difficulty="medium",
    load_level=300,
    max_steps=12,
    starting_schema_sql=STARTING_SCHEMA.strip(),
    target_schema_ddl=TARGET_SCHEMA.strip(),
    seed_data_sql=SEED_DATA.strip(),
    grade_fn=grade_medium,
)
