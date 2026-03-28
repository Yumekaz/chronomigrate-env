from server.lock_analyzer import analyze_lock


def test_add_column_with_default_is_safe():
    profile = analyze_lock("ALTER TABLE users ADD COLUMN email VARCHAR(255) DEFAULT NULL;")
    assert profile.lock_ticks == 0
    assert profile.failure_rate == 0.0


def test_drop_table_is_destructive():
    profile = analyze_lock("DROP TABLE users;")
    assert profile.destroys_data is True
    assert profile.failure_rate == 1.0


def test_create_index_concurrently_is_safe():
    profile = analyze_lock(
        "CREATE INDEX CONCURRENTLY idx_orders_user_id ON orders (user_id);"
    )
    assert profile.lock_ticks == 0


def test_rename_column_is_logged_as_exclusive():
    profile = analyze_lock("ALTER TABLE users RENAME COLUMN id TO user_id;")
    assert profile.lock_type == "ACCESS_EXCLUSIVE"
    assert profile.lock_ticks == 3


def test_add_foreign_key_is_more_expensive_than_simple_add_column():
    profile = analyze_lock(
        "ALTER TABLE orders ADD CONSTRAINT fk_orders_users FOREIGN KEY (user_id) REFERENCES users(id);"
    )
    assert profile.lock_type == "SHARE_ROW_EXCLUSIVE"
    assert profile.lock_ticks == 4


def test_invalid_sql_returns_no_lock():
    profile = analyze_lock("THIS IS NOT SQL")
    assert profile.lock_ticks == 0
    assert profile.failure_rate == 0.0
