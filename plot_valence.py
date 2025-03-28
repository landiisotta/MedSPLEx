import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from yaml import safe_load
import argparse

DEID_ORDER_MAP = {
    "dressed": 0,
    "good_insight": 1,
    "knowledgeable": 2,
    "motivated": 3,
    "support_system": 4,
    "angry": 5,
    "argumentative": 6,
    "hysterical": 7,
    "insist": 8,
    "supposedly": 9,
    "active": 10,
    "adherence": 11,
    "difficult": 12,
    "dirty": 13,
    "educated": 14,
    "employed": 15,
    "fail": 16,
    "informed": 17,
    "inspiring": 18,
    "involved": 19,
    "kind": 20,
    "lazy": 21,
    "managed": 22,
    "resist": 23,
    "skilled": 24,
    "unreliable": 25,
    "engaged": 26,
    "groomed": 27,
    "healthy": 28,
    "compliant": 29,
    "happy": 30,
    "willing": 31,
    "aggressive": 32,
    "agitated": 33,
    "challenging": 34,
    "combative": 35,
    "irritable": 36,
    "refuse": 37,
    "adamant": 38,
    "anxious": 39,
    "apparently": 40,
    "appears_to_be": 41,
    "believes": 42,
    "charming": 43,
    "claim": 44,
    "complain": 45,
    "confront": 46,
    "cordial": 47,
    "defensive": 48,
    "deny": 49,
    "disciplined": 50,
    "endorse": 51,
    "exaggerate": 52,
    "friendly": 53,
    "noncooperative": 54,
    "quotations": 55,
    "reports": 56,
    "say": 57,
    "sedentary": 58,
    "tell": 59,
    "unkempt": 60,
}

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

CONFIG_PATH = "config/keywords_updated.yaml"
OUTPUT_FNAME = "dotplot_deid"


def get_parser():
    """Returns the parser for all the args for run_ddqc function used via script/cli."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        "-i",
        default="config/mimic_keywords.yaml",
        help="Path of the directory that contains the words config.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default=".",
        help="Path of the directory where to save the plot.",
    )
    parser.add_argument(
        "--plot-fname",
        "-n",
        default="dotplot_died",
        help="Name of the the plot.",
    )
    return parser


def scale_score(df, score: float):
    scaler = MinMaxScaler(feature_range=(-5, -1))
    return scaler.fit_transform(np.array(df[score]).reshape(-1, 1))


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return safe_load(f)


def dotplot_words_valence(config_path, output_path, plot_fname, save=True):
    valence_df = pd.DataFrame(load_config(config_path)).T
    valence_df["pattern"] = pd.Categorical(
        valence_df.index, categories=sorted(valence_df.index.unique()), ordered=True
    )
    
    valence_df["pattern_index"] = valence_df["pattern"].map(DEID_ORDER_MAP)

    valence_df["color_privileging"] = valence_df["privileging_score"].apply(
        lambda x: COLOR_PRIV if x < 3 else COLOR_UNLABELED
    )
    valence_df["color_stigmatizing"] = valence_df["stigmatizing_score"].apply(
        lambda x: COLOR_STIG if x < 3 else COLOR_UNLABELED
    )

    valence_df["privileging_score_mapped"] = list(
        np.abs(scale_score(valence_df, "privileging_score")).flatten()
    )
    valence_df["stigmatizing_score_mapped"] = list(
        scale_score(valence_df, "stigmatizing_score").flatten()
    )

    plt.figure(figsize=(15, 6))

    pattern_words = []

    for category in VALENCE_CATEGORIES:
        subset = valence_df[valence_df["valence"] == category]
        subset_x = subset["pattern_index"]
        subset_priv_score = subset["privileging_score_mapped"]
        subset_stig_score = subset["stigmatizing_score_mapped"]
        pattern_words.append(list(subset_x))

        if category in ["undefined", "stigmatizing", "ambiguous_stigmatizing"]:
            plt.scatter(
                subset_x,
                subset_priv_score,
                edgecolor=subset["color_privileging"],
                facecolor="none",
                label=f"{category}",
            )
        elif category == "ambiguous_privileging":
            plt.scatter(
                subset_x,
                subset_priv_score,
                label=f"{category}",
                c=subset["color_privileging"],
            )
            plt.scatter(
                subset_x,
                np.zeros_like(subset_priv_score),
                color=COLOR_NEUT,
                label=f"{category}",
            )
            for i in range(0, subset.shape[0]):
                plt.plot(
                    [subset_x[i], subset_x[i]],
                    [
                        subset_priv_score[i],
                        np.zeros_like(subset_stig_score[i]),
                    ],
                    color="black",
                )

        elif category == "neutral":
            plt.scatter(
                subset_x,
                subset_priv_score,
                edgecolor=subset["color_privileging"],
                facecolor="none",
                label=f"{category}",
            )
            plt.scatter(
                subset_x,
                np.zeros_like(subset_priv_score),
                color=COLOR_NEUT,
                label=f"{category}",
            )
        else:
            plt.scatter(
                subset_x,
                subset_priv_score,
                label=f"{category}",
                c=subset["color_privileging"],
            )

            if category == "ambiguous_bivalent":
                for i in range(0, subset.shape[0]):
                    plt.plot(
                        [subset_x[i], subset_x[i]],
                        [
                            subset_stig_score[i],
                            subset_priv_score[i],
                        ],
                        color="black",
                    )
                    plt.scatter(
                        subset_x,
                        np.zeros_like(subset_priv_score),
                        color=COLOR_NEUT,
                        label=f"{category}",
                    )

    for category in VALENCE_CATEGORIES:
        subset = valence_df[valence_df["valence"] == category]
        subset_x = subset["pattern_index"]
        subset_priv_score = subset["privileging_score_mapped"]
        subset_stig_score = subset["stigmatizing_score_mapped"]
        if category in ["undefined", "privileging", "ambiguous_privileging"]:
            plt.scatter(
                subset_x,
                subset_stig_score,
                edgecolor=subset["color_stigmatizing"],
                facecolor="none",
                label=f"{category}",
            )
        elif category == "ambiguous_stigmatizing":
            plt.scatter(
                subset_x,
                subset_stig_score,
                label=f"{category}",
                c=subset["color_stigmatizing"],
            )
            plt.scatter(
                subset_x,
                np.zeros_like(subset_stig_score),
                color=COLOR_NEUT,
                label=f"{category}",
            )
            for i in range(0, subset.shape[0]):
                plt.plot(
                    [subset_x[i], subset_x[i]],
                    [
                        subset_stig_score[i],
                        np.zeros_like(subset_priv_score[i]),
                    ],
                    color="black",
                )

        elif category == "neutral":
            plt.scatter(
                subset_x,
                subset_stig_score,
                edgecolor=subset["color_stigmatizing"],
                facecolor="none",
                label=f"{category}",
            )

        else:
            plt.scatter(
                subset_x,
                subset_stig_score,
                label=f"{category}",
                c=subset["color_stigmatizing"],
            )
    plt.xticks(
        ticks=range(len(DEID_ORDER_MAP.keys())),
        labels=DEID_ORDER_MAP.keys(),
        rotation=90,
    )
    plt.axhline(0, color="#4C4C4C")
    plt.xlabel("pattern")
    plt.ylabel("")
    plt.xticks(rotation=90)
    plt.yticks(
        ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        labels=["1", "2", "3", "4", "5", "Neutral", "5", "4", "3", "2", "1"],
    )
    plt.tight_layout()
    ax = plt.gca()

    ax.text(
        0.01,
        0.75,
        "Privileging",
        transform=ax.transAxes,
        rotation=90,
        fontsize=10,
        va="center",
        ha="center",
        color=COLOR_PRIV,
    )

    ax.text(
        0.01,
        0.25,
        "Stigmatizing",
        transform=ax.transAxes,
        rotation=90,
        fontsize=10,
        va="center",
        ha="center",
        color=COLOR_STIG,
    )

    if save:
        plt.savefig(f"{output_path}/{plot_fname}.png")
    plt.show()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    dotplot_words_valence(
        config_path=args.config_path,
        output_path=args.output_path,
        plot_fname = args.plot_fname
    )
