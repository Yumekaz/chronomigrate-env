import sqlite3

from server.schema_grader import compute_data_hash, compute_schema_match


def test_schema_match_rewards_target_features():
    current = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL
    );
    """
    target = """
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        email VARCHAR(255) DEFAULT NULL
    );
    """
    assert 0.0 < compute_schema_match(current, target) < 1.0


def test_schema_match_handles_foreign_keys():
    current = """
    CREATE TABLE users (id SERIAL PRIMARY KEY);
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    target = """
    CREATE TABLE users (user_id SERIAL PRIMARY KEY);
    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """
    assert compute_schema_match(current, target) < 1.0


def test_compute_data_hash_is_deterministic_for_sqlite():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, username TEXT)")
    conn.execute("INSERT INTO users (id, username) VALUES (1, 'alice'), (2, 'bob')")
    first = compute_data_hash(conn)
    second = compute_data_hash(conn)
    assert first == second
