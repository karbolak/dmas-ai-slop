import doces


def main():
    model = doces.Opinion_dynamics(
        vertex_count=2,
        edges=[
            (0, 1),
            (1, 0),
        ],
        directed=True,
        verbose=False,
    )

    # Only user 0 can activate.
    model.set_activity_weights(
        [1.0, 0.0]
    )

    model.simulate_dynamics(
        number_of_iterations=1000,
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

    origins = model.post_origin_user_ids
    received = model.post_received_counts

    assert len(origins) == len(received)
    assert len(origins) > 0

    background = [
        i
        for i, origin in enumerate(origins)
        if origin == -1
    ]

    experimental = [
        i
        for i, origin in enumerate(origins)
        if origin >= 0
    ]

    print("Total posts:", len(origins))
    print("Background posts:", len(background))
    print("Experimental posts:", len(experimental))

    # Two users × feed size 1.
    assert len(background) == 2

    # Only user 0 can activate, so every simulation-created
    # post must have user 0 as its original creator.
    assert experimental

    assert all(
        origins[i] == 0
        for i in experimental
    )

    # With mu=1 the initialization posts are never reposted,
    # so initialization itself must not create exposure events.
    assert all(
        received[i] == 0
        for i in background
    )

    total_exposure = sum(
        received[i]
        for i in experimental
    )

    print(
        "Experimental exposure events:",
        total_exposure,
    )

    stats = model.get_cascade_stats_dict()

    assert stats["origin_user_id"] == origins
    assert stats["received_count"] == received

    print("✓ Post origins are exposed correctly.")
    print("✓ Background posts use origin -1.")
    print("✓ Cascade stats expose provenance metadata.")
    print("✓ Content provenance checks passed.")


if __name__ == "__main__":
    main()