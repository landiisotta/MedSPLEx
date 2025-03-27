import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from yaml import safe_load

VALENCE_CATEGORIES = [
    "privileging",
    "stigmatizing",
    "neutral",
    "ambiguous_privileging",
    "ambiguous_bivalent",
    "ambiguous_stigmatizing",
    "undefined",
]
COLOR_PRIV = "#84B47C"
COLOR_STIG = "#FC6B68"
COLOR_NEUT = "#679CDF"
COLOR_UNLABELED = "#040404"


def scale_score(score):
    return scaler.fit_transform(np.array(valence_df[score]).reshape(-1, 1))


with open("config/keywords_updated.yaml", "r") as f:
    config_file = safe_load(f)

valence_df = pd.DataFrame(config_file).T
valence_df["pattern"] = valence_df.index

valence_df["color_privileging"] = valence_df["privileging_score"].apply(lambda x: COLOR_PRIV if x < 3 else COLOR_UNLABELED)
valence_df["color_stigmatizing"] = valence_df["stigmatizing_score"].apply(lambda x: COLOR_STIG if x < 3 else COLOR_UNLABELED)


scaler = MinMaxScaler(feature_range=(-5, -1))

valence_df["privileging_score_mapped"] = list(
    np.abs(scale_score("privileging_score")).flatten()
)
valence_df["stigmatizing_score_mapped"] = list(
    scale_score("stigmatizing_score").flatten()
)


plt.figure(figsize=(15, 6))

for category in VALENCE_CATEGORIES:
    subset = valence_df[valence_df["valence"] == category]
    subset_pattern = subset["pattern"]
    subset_priv_score = subset["privileging_score_mapped"]
    subset_stig_score = subset["stigmatizing_score_mapped"]

    if category in ["undefined", "stigmatizing", "ambiguous_stigmatizing"]:
        plt.scatter(
            subset_pattern,
            subset_priv_score,
            edgecolor=subset["color_privileging"],
            facecolor="none",
            label=f"{category}",
        )
    elif category == "ambiguous_privileging":
        plt.scatter(
            subset_pattern,
            subset_priv_score,
            label=f"{category}",
            c=subset["color_privileging"],
        )
        plt.scatter(
            subset_pattern,
            np.zeros_like(subset_priv_score),
            color=COLOR_NEUT,
            label=f"{category}",
        )
        for i in range(0, subset.shape[0]):
            plt.plot(
                [subset_pattern[i], subset_pattern[i]],
                [
                    subset_priv_score[i],
                    np.zeros_like(subset_stig_score[i]),
                ],
                color="black",
            )

    elif category == "neutral":
        plt.scatter(
            subset_pattern,
            subset_priv_score,
            edgecolor=subset["color_privileging"],
            facecolor="none",
            label=f"{category}",
        )
        plt.scatter(
            subset_pattern,
            np.zeros_like(subset_priv_score),
            color=COLOR_NEUT,
            label=f"{category}",
        )
    else:
        plt.scatter(
            subset_pattern,
            subset_priv_score,
            label=f"{category}",
            c=subset["color_privileging"],
        )

        if category == "ambiguous_bivalent":
            for i in range(0, subset.shape[0]):
                plt.plot(
                    [subset_pattern[i], subset_pattern[i]],
                    [
                        subset_stig_score[i],
                        subset_priv_score[i],
                    ],
                    color="black",
                )
                plt.scatter(
                    subset_pattern,
                    np.zeros_like(subset_priv_score),
                    color=COLOR_NEUT,
                    label=f"{category}",
                )

for category in VALENCE_CATEGORIES:
    subset = valence_df[valence_df["valence"] == category]
    subset_pattern = subset["pattern"]
    subset_priv_score = subset["privileging_score_mapped"]
    subset_stig_score = subset["stigmatizing_score_mapped"]
    if category in ["undefined", "privileging", "ambiguous_privileging"]:
        plt.scatter(
            subset_pattern,
            subset_stig_score,
            edgecolor=subset["color_stigmatizing"],
            facecolor="none",
            label=f"{category}",
        )
    elif category == "ambiguous_stigmatizing":
        plt.scatter(
            subset_pattern,
            subset_stig_score,
            label=f"{category}",
            c=subset["color_stigmatizing"],
        )
        plt.scatter(
            subset_pattern,
            np.zeros_like(subset_stig_score),
            color=COLOR_NEUT,
            label=f"{category}",
        )
        for i in range(0, subset.shape[0]):
            plt.plot(
                [subset_pattern[i], subset_pattern[i]],
                [
                    subset_stig_score[i],
                    np.zeros_like(subset_priv_score[i]),
                ],
                color="black",
            )

    elif category == "neutral":
        plt.scatter(
            subset_pattern,
            subset_stig_score,
            edgecolor=subset["color_stigmatizing"],
            facecolor="none",
            label=f"{category}",
        )

    else:
        plt.scatter(
            subset_pattern,
            subset_stig_score,
            label=f"{category}",
            c=subset["color_stigmatizing"],
        )


plt.axhline(0, color='#4C4C4C')


plt.xlabel("pattern")
plt.ylabel("")

plt.xticks(rotation=90)
plt.yticks(
    ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    labels=["1", "2", "3", "4", "5", "Neutral", "5", "4", "3", "2", "1"],
)
plt.tight_layout()
ax = plt.gca()

ax.text(0.01, 0.75, 'Privileging',
        transform=ax.transAxes, rotation=90,
        fontsize=10,  va='center', ha='center', color=COLOR_PRIV)

ax.text(0.01, 0.25, 'Stigmatizing',
        transform=ax.transAxes, rotation=90,
        fontsize=10,  va='center', ha='center', color=COLOR_STIG)


plt.savefig("plot_tmp.png")
plt.show()
