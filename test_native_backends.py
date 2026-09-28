import cpmpy as cp
from cpmpy import Model, intvar, boolvar, AllDifferent, Table


SOLVERS = ["minicp_native", "maxicp_native"]


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"[OK] {name}")


def test_table(solver):
    x = cp.intvar(1, 3, name="x")
    y = cp.intvar(1, 3, name="y")

    table = [
        [1, 2],
        [2, 3],
    ]

    model = cp.Model(
        cp.Table([x, y], table)
    )

    assert model.solve(solver=solver)
    assert (x.value(), y.value()) in [(1, 2), (2, 3)]

    print("table:", x.value(), y.value())


def run_backend(solver):
    print()
    print("=" * 60)
    print(f"Testing {solver}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Basic satisfaction
    # ---------------------------------------------------------
    x = intvar(0, 10, name="x")
    y = intvar(0, 10, name="y")

    m = Model(
        x + y == 10,
        x != y,
    )

    result = m.solve(solver=solver)

    check(f"{solver}: basic SAT", result)
    check(f"{solver}: basic sum", x.value() + y.value() == 10)
    check(f"{solver}: basic inequality", x.value() != y.value())

    # ---------------------------------------------------------
    # 2. UNSAT
    # ---------------------------------------------------------
    x = intvar(0, 10, name="x")

    m = Model(
        x == 3,
        x == 4,
    )

    result = m.solve(solver=solver)

    check(f"{solver}: UNSAT", result is False)

    # ---------------------------------------------------------
    # 3. Comparisons
    # ---------------------------------------------------------
    x = intvar(0, 20, name="x")

    m = Model(
        x == 10,
        x != 9,
        x <= 10,
        x < 11,
        x >= 10,
        x > 9,
    )

    result = m.solve(solver=solver)

    check(f"{solver}: comparisons", result and x.value() == 10)

    # ---------------------------------------------------------
    # 4. Weighted linear
    # ---------------------------------------------------------
    x = intvar(0, 10, name="x")
    y = intvar(0, 10, name="y")

    m = Model(
        2 * x + 3 * y == 13,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: weighted linear",
        result and 2 * x.value() + 3 * y.value() == 13,
    )

    # ---------------------------------------------------------
    # 5. Linear inequalities
    # ---------------------------------------------------------
    x = intvar(0, 10, name="x")
    y = intvar(0, 10, name="y")

    m = Model(
        2 * x + y <= 10,
        x + 3 * y >= 8,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: linear inequalities",
        result
        and 2 * x.value() + y.value() <= 10
        and x.value() + 3 * y.value() >= 8,
    )

    # ---------------------------------------------------------
    # 6. Boolean variables / negation
    # ---------------------------------------------------------
    a = boolvar(name="a")
    b = boolvar(name="b")
    c = boolvar(name="c")

    m = Model(
        a,
        ~b,
        c == a,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: Boolean variables",
        result
        and bool(a.value())
        and not bool(b.value())
        and bool(c.value()),
    )

    # ---------------------------------------------------------
    # 7. AND / OR
    # ---------------------------------------------------------
    a = boolvar(name="a")
    b = boolvar(name="b")

    m = Model(
        a | b,
        a & ~b,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: AND/OR",
        result and bool(a.value()) and not bool(b.value()),
    )

    # ---------------------------------------------------------
    # 8. Implication
    # ---------------------------------------------------------
    a = boolvar(name="a")
    x = intvar(0, 10, name="x")

    m = Model(
        a,
        a.implies(x >= 7),
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: implication",
        result and bool(a.value()) and x.value() >= 7,
    )

    # ---------------------------------------------------------
    # 9. Reified equality
    # ---------------------------------------------------------
    b = boolvar(name="b")
    x = intvar(0, 10, name="x")

    m = Model(
        b == (x == 4),
        b,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: reified equality",
        result and bool(b.value()) and x.value() == 4,
    )

    # ---------------------------------------------------------
    # 10. Reified inequality
    # ---------------------------------------------------------
    b = boolvar(name="b")
    x = intvar(0, 10, name="x")

    m = Model(
        b == (x <= 5),
        ~b,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: reified <=",
        result and not bool(b.value()) and x.value() > 5,
    )

    # ---------------------------------------------------------
    # 11. AllDifferent
    # ---------------------------------------------------------
    xs = intvar(1, 3, shape=3, name="xs")

    m = Model(
        AllDifferent(xs),
        sum(xs) == 6,
    )

    result = m.solve(solver=solver)

    values = [v.value() for v in xs]

    check(
        f"{solver}: AllDifferent",
        result and sorted(values) == [1, 2, 3],
    )

    # ---------------------------------------------------------
    # 12. Minimize
    # ---------------------------------------------------------
    x = intvar(0, 100, name="x")
    y = intvar(0, 100, name="y")

    m = Model(
        x + y == 100,
        x >= 30,
        minimize=x,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: minimize",
        result and x.value() == 30 and m.objective_value() == 30,
    )

    # ---------------------------------------------------------
    # 13. Maximize
    # ---------------------------------------------------------
    x = intvar(0, 100, name="x")
    y = intvar(0, 100, name="y")

    m = Model(
        x + y == 100,
        x <= 80,
        maximize=x,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: maximize",
        result and x.value() == 80 and m.objective_value() == 80,
    )

    # ---------------------------------------------------------
    # 14. Linear objective
    # ---------------------------------------------------------
    x = intvar(0, 10, name="x")
    y = intvar(0, 10, name="y")

    m = Model(
        x + y >= 5,
        minimize=2 * x + 3 * y,
    )

    result = m.solve(solver=solver)

    check(
        f"{solver}: linear objective",
        result and m.objective_value() == 10,
    )

    # ---------------------------------------------------------
    # 15. Incremental optimization
    # ---------------------------------------------------------
    x = intvar(0, 100, name="x")

    m = Model(minimize=x)

    s = cp.SolverLookup.get(solver, m)

    first = s.solve()
    first_value = x.value()

    s += x >= 40

    second = s.solve()
    second_value = x.value()

    check(
        f"{solver}: incremental optimization",
        first
        and second
        and first_value == 0
        and second_value == 40,
    )

    # ---------------------------------------------------------
    # 16. Table
    # ---------------------------------------------------------
    print()
    print(f"Running test_table for {solver}")
    test_table(solver)

    print()
    print(f"{solver} validated")


for solver in SOLVERS:
    run_backend(solver)


print()
print("=" * 60)
print("BOTH NATIVE BACKENDS VALIDATED")
print("=" * 60)