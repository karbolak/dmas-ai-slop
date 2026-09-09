import pandas as pd

df = pd.read_csv(
    "results/oliveira_fixed_state_seed_sweep/runs.csv"
)

mu1 = (
    df[df["mu"] == 1.0]
    .set_index("seed")
)

mu01 = (
    df[df["mu"] == 0.1]
    .set_index("seed")
)

candidates = mu01[
    (mu1["bimodality_coefficient"] < 5 / 9)
    & (mu01["bimodality_coefficient"] > 5 / 9)
]

print(
    candidates[
        [
            "bimodality_coefficient",
            "positive_fraction",
            "negative_fraction",
            "std_opinion",
            "opinion_sign_assortativity",
        ]
    ]
    .sort_values(
        "bimodality_coefficient",
        ascending=False,
    )
)