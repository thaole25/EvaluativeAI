import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

from collections import defaultdict
import math

from preprocessing import params


def CI(mean, std, count):
    ci95_hi = mean + 1.96 * std / math.sqrt(count)
    ci95_lo = mean - 1.96 * std / math.sqrt(count)
    return ci95_lo, ci95_hi


def get_mean_std_measure(method, measure_name, df, df_stats):
    method_mean, method_std = round(df[method + "_" + measure_name].mean(), 2), round(
        df[method + "_" + measure_name].std(), 2
    )
    df_stats["{}_{}_mean".format(method, measure_name)].append(method_mean)
    df_stats["{}_{}_std".format(method, measure_name)].append(method_std)
    s = "$" + str(method_mean) + " \pm " + str(method_std) + "$"
    df_stats["{}_{}".format(method, measure_name)].append(s)


def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s


NUM_SEEDS = 20
MODELS = [
    "resnet50",
    "resneXt50",
    "resnet152",
]
CONCEPTS1 = list(np.arange(4, 13))
CONCEPTS2 = list(np.arange(5, 45, 5))
ALL_CONCEPTS = list(np.arange(4, 13)) + list(np.arange(15, 45, 5))

LRS = [0.01]
NSAMPLES = [50]
PCBM_CLFS = ["sgd", "ridge"]

FEATURE_TYPES = ["mean"]
IS_TRAIN_CLFS = [False, True]
ICE_CLFS = ["ridge", "gnb", "Not-Applicable"]
REDUCERS = ["NMF", "PCA"]
WOE_CLFS = ["original"]
SELECTED_MEASURES = ["acc", "sensitivity", "precision", "f1_score"]


cnn_df = pd.read_csv(params.BACKBONE_TRAIN_FILE)
cnn_df["model"].replace("resnext50", "resneXt50", inplace=True)
cnn_stats = defaultdict(list)

for model in MODELS:
    acc_col = cnn_df[cnn_df["model"] == model]
    acc_count = acc_col.shape[0]
    if acc_count == 0:
        continue
    cnn_stats["model"].append(model)
    cnn_stats["count"].append(acc_count)
    acc_mean = acc_col["test_acc"].mean()
    acc_std = acc_col["test_acc"].std()
    acc = "$" + str(round(acc_mean, 2)) + " \pm " + str(round(acc_std, 2)) + "$"
    cnn_stats["mean accuracy"].append(acc_mean)
    cnn_stats["std accuracy"].append(acc_std)
    cnn_stats["accuracy"].append(acc)
    ci95_lo, ci95_hi = CI(acc_mean, acc_std, acc_count)
    ci95 = "[" + str(round(ci95_lo, 2)) + ", " + str(round(ci95_hi, 2)) + "]"
    cnn_stats["95\% CI"].append(ci95)
    cnn_stats["max"].append(acc_col["test_acc"].max())
    time_mean = acc_col["test_time_cost"].mean()
    time_std = acc_col["test_time_cost"].std()
    cnn_stats["time cost (s)"].append(
        "$" + str(round(time_mean, 2)) + " \pm " + str(round(time_std, 2)) + "$"
    )

cnn_stats_df = pd.DataFrame(cnn_stats)
cnn_stats_df.to_csv(params.STATS_CNN_FILE)
print(
    cnn_stats_df[["model", "accuracy", "95\% CI", "time cost (s)"]].to_latex(
        index=False, bold_rows=True
    )
)

ice_df = pd.read_csv(params.ICE_RESULT_FILE)
ice_woe_df = pd.read_csv(params.ICE_WOE_RESULT_FILE)
pcbm_7pt_df = pd.read_csv(params.PCBM_CONCEPT_7PT_FILE)
pcbm_df = pd.read_csv(params.PCBM_RESULT_FILE)
pcbm_woe_df = pd.read_csv(params.PCBM_WOE_RESULT_FILE)

ice_df["model"].replace("resnext50", "resneXt50", inplace=True)
ice_df["ice_clf"].replace(float("nan"), "Not-Applicable", inplace=True)
ice_woe_df["model"].replace("resnext50", "resneXt50", inplace=True)
ice_woe_df["ice_clf"].replace(float("nan"), "Not-Applicable", inplace=True)
pcbm_7pt_df["model"].replace("resnext50", "resneXt50", inplace=True)
pcbm_df["model"].replace("resnext50", "resneXt50", inplace=True)
pcbm_woe_df["model"].replace("resnext50", "resneXt50", inplace=True)


# Handle ICE
ice_concept_stats = defaultdict(list)
iceclf_idx = 0
for model in MODELS:
    for concept in ALL_CONCEPTS:
        for feature_type in FEATURE_TYPES:
            for reducer in REDUCERS:
                for is_train_clf in IS_TRAIN_CLFS:
                    for ice_clf in ICE_CLFS:
                        concept_df = ice_df[
                            (ice_df["model"] == model)
                            & (ice_df["concept"] == concept)
                            & (ice_df["feature_type"] == feature_type)
                            & (ice_df["reducer"] == reducer)
                            & (ice_df["is_train_clf"] == is_train_clf)
                            & (ice_df["ice_clf"] == ice_clf)
                        ]
                        if concept_df.shape[0] == 0:
                            continue
                        woe_df = ice_woe_df[
                            (ice_woe_df["model"] == model)
                            & (ice_woe_df["concept"] == concept)
                            & (ice_woe_df["feature_type"] == feature_type)
                            & (ice_woe_df["reducer"] == reducer)
                            & (ice_woe_df["woe_clf"] == WOE_CLFS[0])
                        ]
                        acc_count = max(concept_df.shape[0], woe_df.shape[0])
                        ice_concept_stats["backbone_model"].append(model)
                        ice_concept_stats["is_train_clf"].append(is_train_clf)
                        ice_concept_stats["ice_clf"].append(ice_clf)
                        ice_concept_stats["reducer"].append(reducer)
                        ice_concept_stats["ICE no concepts"].append(concept)
                        ice_concept_stats["ice count"].append(acc_count)
                        if (
                            concept < 10
                            and model == "resneXt50"
                            and not is_train_clf
                            and reducer == "NMF"
                        ):
                            print(model, concept, is_train_clf, reducer)
                            print(
                                "ice vs. woe",
                                ttest_ind(
                                    concept_df["ice_f1_score"], woe_df["woe_f1_score"]
                                ),
                                cohend(
                                    concept_df["ice_f1_score"], woe_df["woe_f1_score"]
                                ),
                            )
                            print(
                                "original vs. woe",
                                ttest_ind(
                                    cnn_df[cnn_df["model"] == model]["test_f1_score"],
                                    woe_df["woe_f1_score"],
                                ),
                                cohend(
                                    cnn_df[cnn_df["model"] == model]["test_f1_score"],
                                    woe_df["woe_f1_score"],
                                ),
                            )
                            print(
                                "original vs. ice",
                                ttest_ind(
                                    concept_df["ice_f1_score"],
                                    cnn_df[cnn_df["model"] == model]["test_f1_score"],
                                ),
                                cohend(
                                    concept_df["ice_f1_score"],
                                    cnn_df[cnn_df["model"] == model]["test_f1_score"],
                                ),
                            )
                        for measure in SELECTED_MEASURES:
                            get_mean_std_measure(
                                method="test",
                                measure_name=measure,
                                df=cnn_df[cnn_df["model"] == model],
                                df_stats=ice_concept_stats,
                            )
                            get_mean_std_measure(
                                method="ice",
                                measure_name=measure,
                                df=concept_df,
                                df_stats=ice_concept_stats,
                            )
                            get_mean_std_measure(
                                method="woe",
                                measure_name=measure,
                                df=woe_df,
                                df_stats=ice_concept_stats,
                            )

# Handle PCBM
pcbm_7pt_stats = defaultdict(list)
for model in MODELS:
    for lr in LRS:
        for nsample in NSAMPLES:
            pt7_df = pcbm_7pt_df[
                (pcbm_7pt_df["model"] == model)
                & (pcbm_7pt_df["lr"] == lr)
                & (pcbm_7pt_df["concept_samples"] == nsample)
            ]
            acc_count = pt7_df.shape[0]
            if acc_count == 0:
                continue
            pcbm_7pt_stats["backbone_model"].append(model)
            model_mean = round(
                cnn_stats_df[cnn_stats_df["model"] == model]["mean accuracy"].iloc[0], 2
            )
            model_std = round(
                cnn_stats_df[cnn_stats_df["model"] == model]["std accuracy"].iloc[0], 2
            )
            model_mean = ("%f" % model_mean).rstrip("0").rstrip(".")
            model_std = ("%f" % model_std).rstrip("0").rstrip(".")
            pcbm_7pt_stats["backbone_model_mean"].append(model_mean)
            pcbm_7pt_stats["backbone_model_std"].append(model_std)
            pcbm_7pt_stats["learning rate"].append(("%f" % lr).rstrip("0").rstrip("."))
            pcbm_7pt_stats["7pt no training samples"].append(nsample)
            pcbm_7pt_stats["PCBM no concepts"].append(pt7_df["no_concepts"].iloc[0])

            method = "7pt"
            method_mean, method_std = round(
                pt7_df[method + "_test_acc"].mean(), 2
            ), round(pt7_df[method + "_test_acc"].std(), 2)
            method_ci95_lo, method_ci95_hi = CI(method_mean, method_std, acc_count)
            pcbm_7pt_stats[method.upper() + " 95\% CI"].append(
                [round(method_ci95_lo, 2), round(method_ci95_hi, 2)]
            )
            pcbm_7pt_stats[method + "_mean"].append(method_mean)
            pcbm_7pt_stats[method + "_std"].append(method_std)
            pcbm_7pt_stats[method.upper() + " Accuracy"].append(
                "$" + str(method_mean) + " \pm " + str(method_std) + "$"
            )
            pcbm_7pt_stats[method + "_max"].append(
                round(pt7_df[method + "_test_acc"].max(), 2)
            )


pcbm_concept_stats = defaultdict(list)
for clf_layer in PCBM_CLFS:
    for model in MODELS:
        for nsample in NSAMPLES:
            for lr in LRS:
                concept_df = pcbm_df[
                    (pcbm_df["model"] == model)
                    & (pcbm_df["lr"] == lr)
                    & (pcbm_df["concept_samples"] == nsample)
                    & (pcbm_df["pcbm_classifier"] == clf_layer)
                ]
                woe_df = pcbm_woe_df[
                    (pcbm_woe_df["model"] == model)
                    & (pcbm_woe_df["lr"] == lr)
                    & (pcbm_woe_df["concept_samples"] == nsample)
                    & (pcbm_woe_df["pcbm_classifier"] == clf_layer)
                ]
                if concept_df.shape[0] != NUM_SEEDS:
                    acc_count = concept_df.shape[0]
                elif woe_df.shape[0] != NUM_SEEDS:
                    acc_count = woe_df.shape[0]
                else:
                    acc_count = NUM_SEEDS
                if acc_count == 0:
                    continue
                pcbm_concept_stats["backbone_model"].append(model)
                pcbm_concept_stats["pcbm_classifier"].append(clf_layer)
                pcbm_concept_stats["learning rate"].append(
                    ("%f" % lr).rstrip("0").rstrip(".")
                )
                pcbm_concept_stats["7pt no training samples"].append(nsample)
                pcbm_concept_stats["PCBM no concepts"].append(
                    concept_df["no_concepts"].iloc[0]
                )
                for measure in SELECTED_MEASURES:
                    get_mean_std_measure(
                        method="test",
                        measure_name=measure,
                        df=cnn_df[cnn_df["model"] == model],
                        df_stats=pcbm_concept_stats,
                    )
                    get_mean_std_measure(
                        method="pcbm",
                        measure_name=measure,
                        df=concept_df,
                        df_stats=pcbm_concept_stats,
                    )
                    get_mean_std_measure(
                        method="woe",
                        measure_name=measure,
                        df=woe_df,
                        df_stats=pcbm_concept_stats,
                    )
                pcbm_concept_stats["pcbm count"].append(acc_count)

ice_stats_df = pd.DataFrame(ice_concept_stats)
ice_stats_df.to_csv(params.STATS_ICE_FILE)

pcbm_7pt_stats_df = pd.DataFrame(pcbm_7pt_stats)
pcbm_7pt_stats_df.to_csv(params.STATS_7PT_FILE)
# print(pcbm_7pt_stats_df[['backbone_model', 'backbone_model_mean', 'learning rate', '7pt no training samples', 'PCBM no concepts',
#                          '7PT Accuracy', '7PT 95\% CI']].to_latex(index=False, bold_rows=True))

pcbm_stats_df = pd.DataFrame(pcbm_concept_stats)
pcbm_stats_df["learning rate"] = pcbm_stats_df["learning rate"].astype(float)
pcbm_stats_df.to_csv(params.STATS_PCBM_FILE)

# PRINT TO LATEX
SUBMODELS_ICE = ["test", "ice", "woe"]
SUBMODELS_PCBM = ["pcbm", "woe"]
latex_df = pd.DataFrame(columns=["Model", "Precision", "Recall", "F1-Score"])
for model in MODELS:
    ridx = 0
    selected_ice_df = ice_stats_df[
        (ice_stats_df["backbone_model"] == model)
        & (ice_stats_df["ICE no concepts"] == 8)
        & (ice_stats_df["is_train_clf"] == False)
        & (ice_stats_df["reducer"] == "NMF")
    ]
    selected_ridge_ice_df = ice_stats_df[
        (ice_stats_df["backbone_model"] == model)
        & (ice_stats_df["ICE no concepts"] == 8)
        & (ice_stats_df["reducer"] == "NMF")
        & (ice_stats_df["is_train_clf"] == True)
        & (ice_stats_df["ice_clf"] == "ridge")
    ]
    selected_ice_df_5 = ice_stats_df[
        (ice_stats_df["backbone_model"] == model)
        & (ice_stats_df["ICE no concepts"] == 5)
        & (ice_stats_df["is_train_clf"] == False)
        & (ice_stats_df["reducer"] == "NMF")
    ]
    selected_pcbm_df = pcbm_stats_df[
        (pcbm_stats_df["backbone_model"] == model)
        & (pcbm_stats_df["pcbm_classifier"] == "ridge")
        & (pcbm_stats_df["learning rate"] == 0.01)
        & (pcbm_stats_df["7pt no training samples"] == 50)
    ]
    for i, submodel in enumerate(SUBMODELS_ICE):
        latex_df.loc[ridx + i] = [model + " " + submodel + "(8)"] + selected_ice_df[
            [
                submodel + "_precision",
                submodel + "_sensitivity",
                submodel + "_f1_score",
            ]
        ].values.flatten().tolist()
    ridx = ridx + len(SUBMODELS_ICE)
    latex_df.loc[ridx] = [model + " " + "ridge ice(8)"] + selected_ridge_ice_df[
        [
            "ice_precision",
            "ice_sensitivity",
            "ice_f1_score",
        ]
    ].values.flatten().tolist()
    ridx = ridx + 1
    for i, submodel in enumerate(SUBMODELS_PCBM):
        latex_df.loc[ridx + i] = [model + " " + submodel] + selected_pcbm_df[
            [
                submodel + "_precision",
                submodel + "_sensitivity",
                submodel + "_f1_score",
            ]
        ].values.flatten().tolist()
    ridx = ridx + len(SUBMODELS_PCBM)
    for i, submodel in enumerate(SUBMODELS_ICE):
        latex_df.loc[ridx + i] = [model + " " + submodel + "(5)"] + selected_ice_df_5[
            [
                submodel + "_precision",
                submodel + "_sensitivity",
                submodel + "_f1_score",
            ]
        ].values.flatten().tolist()

    print(latex_df.to_latex(index=False, bold_rows=True))

print("Ablation studies: ICE(12) ridge vs. PCBM")
latex_df = pd.DataFrame(columns=["Model", "Precision", "Recall", "F1-Score"])
for model in MODELS:
    ridx = 0
    selected_ridge_ice_df = ice_stats_df[
        (ice_stats_df["backbone_model"] == model)
        & (ice_stats_df["ICE no concepts"] == 12)
        & (ice_stats_df["reducer"] == "NMF")
        & (ice_stats_df["is_train_clf"] == True)
        & (ice_stats_df["ice_clf"] == "ridge")
    ]
    selected_pcbm_df = pcbm_stats_df[
        (pcbm_stats_df["backbone_model"] == model)
        & (pcbm_stats_df["pcbm_classifier"] == "ridge")
        & (pcbm_stats_df["learning rate"] == 0.01)
        & (pcbm_stats_df["7pt no training samples"] == 50)
    ]
    latex_df.loc[ridx] = [model + " " + "ICE(12)+Ridge"] + selected_ridge_ice_df[
        [
            "ice_precision",
            "ice_sensitivity",
            "ice_f1_score",
        ]
    ].values.flatten().tolist()
    ridx = ridx + 1
    latex_df.loc[ridx + i] = [model + " " + "PCBM"] + selected_pcbm_df[
        [
            "pcbm" + "_precision",
            "pcbm" + "_sensitivity",
            "pcbm" + "_f1_score",
        ]
    ].values.flatten().tolist()
    print(latex_df.to_latex(index=False, bold_rows=True))

print("Ablation studies: different classification layers of ICE")
latex_df = pd.DataFrame(columns=["Model", "Precision", "Recall", "F1-Score"])
for model in MODELS:
    ridx = 0
    selected_ice_df = ice_stats_df[
        (ice_stats_df["backbone_model"] == model)
        & (ice_stats_df["ICE no concepts"] == 7)
        & (ice_stats_df["is_train_clf"] == False)
        & (ice_stats_df["reducer"] == "NMF")
    ]
    selected_ice_df_ridge = ice_stats_df[
        (ice_stats_df["backbone_model"] == model)
        & (ice_stats_df["ICE no concepts"] == 7)
        & (ice_stats_df["reducer"] == "NMF")
        & (ice_stats_df["is_train_clf"] == True)
        & (ice_stats_df["ice_clf"] == "ridge")
    ]
    selected_ice_df_gnb = ice_stats_df[
        (ice_stats_df["backbone_model"] == model)
        & (ice_stats_df["ICE no concepts"] == 7)
        & (ice_stats_df["reducer"] == "NMF")
        & (ice_stats_df["is_train_clf"] == True)
        & (ice_stats_df["ice_clf"] == "gnb")
    ]
    latex_df.loc[ridx] = [model + " " + "ICE(7)"] + selected_ice_df[
        [
            "ice_precision",
            "ice_sensitivity",
            "ice_f1_score",
        ]
    ].values.flatten().tolist()
    ridx = ridx + 1
    latex_df.loc[ridx] = [model + " " + "ICE(7)+Ridge"] + selected_ice_df_ridge[
        [
            "ice_precision",
            "ice_sensitivity",
            "ice_f1_score",
        ]
    ].values.flatten().tolist()
    ridx = ridx + 1
    latex_df.loc[ridx] = [model + " " + "ICE(7)+GNB"] + selected_ice_df_gnb[
        [
            "ice_precision",
            "ice_sensitivity",
            "ice_f1_score",
        ]
    ].values.flatten().tolist()
    ridx = ridx + 1
    latex_df.loc[ridx] = [model + " " + "ICE(7)+WoE"] + selected_ice_df[
        [
            "woe_precision",
            "woe_sensitivity",
            "woe_f1_score",
        ]
    ].values.flatten().tolist()

    print(latex_df.to_latex(index=False, bold_rows=True))

LINESTYLES = ["solid", "dotted", "dashdot", "dashed", ("loosely dotted", (0, (1, 10)))]
ice_groups = ice_stats_df.groupby("backbone_model")[
    [
        "ICE no concepts",
        "test_f1_score_mean",
        "test_f1_score_std",
        "ice_f1_score_mean",
        "ice_f1_score_std",
        "woe_f1_score_mean",
        "woe_f1_score_std",
        "is_train_clf",
        "reducer",
    ]
]


def plot(
    method,
    selected_model_df,
    x_axis,
    x_label,
    y_label,
    x_col,
    backbone,
    img_path,
    reducer,
):
    FONTSIZE = 16
    fig, ax = plt.subplots(figsize=(6, 5))
    df = selected_model_df.copy()
    df = df[(df["reducer"] == reducer)]
    df = df.loc[df[x_col].isin(x_axis)]

    ax.errorbar(
        x_axis,
        df["test_f1_score_mean"],
        df["test_f1_score_std"],
        label="Original " + backbone,
        linestyle=LINESTYLES[0],
        marker="o",
    )

    ax.errorbar(
        x_axis,
        df[method + "_f1_score_mean"],
        df[method + "_f1_score_std"],
        label=method.upper(),
        linestyle=LINESTYLES[1],
        marker="o",
    )
    ax.errorbar(
        x_axis,
        df["woe_f1_score_mean"],
        df["woe_f1_score_std"],
        label=method.upper() + "+WoE",
        linestyle=LINESTYLES[2],
        marker="o",
    )
    ax.legend(loc="lower right", fontsize=FONTSIZE)
    ax.set_ylabel(y_label, fontsize=FONTSIZE)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.set_title("{} using {}".format(method, backbone).upper(), fontsize=FONTSIZE + 2)
    ax.set_ylim(40, 100)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE - 5)
    fig.savefig(img_path)


def plot_reducer(
    method, selected_model_df, x_axis, x_label, y_label, x_col, backbone, img_path
):
    FONTSIZE = 16
    fig, ax = plt.subplots(figsize=(6, 5))
    df = selected_model_df.copy()
    df = df.loc[df[x_col].isin(x_axis)]
    for i, reducer in enumerate(REDUCERS):
        selected_df = df[(df["reducer"] == reducer)]
        ax.errorbar(
            x_axis,
            selected_df[method + "_f1_score_mean"],
            selected_df[method + "_f1_score_std"],
            label=reducer + "-ICE",
            linestyle=LINESTYLES[i],
            marker="o",
        )
    ax.legend(loc="lower right", fontsize=FONTSIZE)
    ax.set_ylabel(y_label, fontsize=FONTSIZE)
    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.set_title(
        "{} using {}".format(method, backbone).upper(),
        fontsize=FONTSIZE + 2,
    )
    ax.set_ylim(40, 100)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE - 5)
    fig.savefig(img_path)


for m in ["resneXt50"]:
    if m in ice_groups.groups.keys():
        ice_df = ice_groups.get_group(m)
        ice_df = ice_df[ice_df["is_train_clf"] == False]
        plot_reducer(
            "ice",
            ice_df,
            x_axis=CONCEPTS2,
            x_label="Number of concepts",
            y_label=None,
            x_col="ICE no concepts",
            backbone=m,
            img_path=params.RESULT_PATH / "ice_{}_by_reducers.png".format(m),
        )
        for _, reducer in enumerate(["NMF"]):
            plot(
                "ice",
                ice_df,
                x_axis=CONCEPTS1,
                x_label="Number of concepts",
                y_label="F1-score (%)",
                x_col="ICE no concepts",
                backbone=m,
                img_path=params.RESULT_PATH
                / "small_ice_{}_{}_by_no_concepts.png".format(m, reducer),
                reducer=reducer,
            )
            plot(
                "ice",
                ice_df,
                x_axis=CONCEPTS2,
                x_label="Number of concepts",
                y_label=None,
                x_col="ICE no concepts",
                backbone=m,
                img_path=params.RESULT_PATH
                / "ice_{}_{}_by_no_concepts.png".format(m, reducer),
                reducer=reducer,
            )
