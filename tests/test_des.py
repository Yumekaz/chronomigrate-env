from server.des_simulator import DiscreteEventSimulator


def test_des_is_deterministic():
    first = DiscreteEventSimulator(task_load_level=100, seed=42).simulate_step(5, 0.2)
    second = DiscreteEventSimulator(task_load_level=100, seed=42).simulate_step(5, 0.2)
    assert first == second


def test_des_reports_zero_downtime_for_zero_failure_rate():
    result = DiscreteEventSimulator(task_load_level=100, seed=42).simulate_step(0, 0.0)
    assert result.downtime_pct == 0.0
    assert result.queries_failed == 0


def test_des_step_seed_matches_v3_formula():
    simulator = DiscreteEventSimulator(task_load_level=100, seed=11)
    assert simulator._step_seed() == 11
    simulator.simulate_step(5, 0.2)
    assert simulator._step_seed() == 11 + 31337


def test_des_uses_lock_ticks_to_scale_background_load():
    simulator = DiscreteEventSimulator(task_load_level=300, seed=7)
    result = simulator.simulate_step(5, 0.0)
    assert result.queries_total == 1500
    assert simulator.tick_counter == 1
