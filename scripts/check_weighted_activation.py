import math

import doces


def make_model():
    return doces.Opinion_dynamics(
        vertex_count=2,
        edges=[
            (0, 1),
            (1, 0),
        ],
        directed=True,
        verbose=False,
    )


def run_with_weights(weights, iterations=100_000):
    model = make_model()

    model.set_activity_weights(
        weights
    )

    model.simulate_dynamics(
        number_of_iterations=iterations,
        phi=0.0,
        mu=1.0,
        posting_filter=0,
        receiving_filter=0,
        b=[-0.25, 0.25],
        feed_size=1,
        rewire=False,
        min_opinion=-1,
        max_opinion=1,
        delta=0.1,
        verbose=False,
        rand_seed=42,
    )

    counts = model.activation_counts

    total = sum(counts)

    fractions = [
        count / total
        for count in counts
    ]

    return counts, fractions


def check_weighted():
    counts, fractions = run_with_weights(
        [9.0, 1.0]
    )

    print("=== Weighted activation ===")
    print(f"Counts:    {counts}")
    print(f"Fractions: {fractions}")

    assert 0.87 <= fractions[0] <= 0.93
    assert 0.07 <= fractions[1] <= 0.13

    print("✓ 9:1 weighting behaves as expected")


def check_uniform():
    counts, fractions = run_with_weights(
        [1.0, 1.0]
    )

    print()
    print("=== Explicit uniform weights ===")
    print(f"Counts:    {counts}")
    print(f"Fractions: {fractions}")

    assert 0.47 <= fractions[0] <= 0.53
    assert 0.47 <= fractions[1] <= 0.53

    print("✓ Uniform weights behave as expected")


def check_validation():
    print()
    print("=== Input validation ===")

    invalid_cases = [
        [-1.0, 1.0],
        [0.0, 0.0],
        [1.0, 2.0, 3.0],
        [math.nan, 1.0],
        [math.inf, 1.0],
    ]

    for weights in invalid_cases:
        model = make_model()

        try:
            model.set_activity_weights(
                weights
            )
        except (ValueError, TypeError, RuntimeError):
            print(
                f"✓ rejected {weights}"
            )
        else:
            raise AssertionError(
                f"Invalid weights were accepted: {weights}"
            )


def main():
    check_weighted()
    check_uniform()
    check_validation()

    print()
    print(
        "✓ Weighted activation checks passed."
    )


if __name__ == "__main__":
    main()