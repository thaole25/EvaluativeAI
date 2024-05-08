import os
import pandas as pd
from torch.utils.data import DataLoader

CONCEPT_TYPE = {
    "vascular_structures": {
        "absent": "ABS",
        "arborizing": "REG",
        "comma": "REG",
        "hairpin": "REG",
        "within regression": "REG",
        "wreath": "REG",
        "dotted": "IR",
        "linear irregular": "IR",
    },
    "pigmentation": {
        "absent": "ABS",
        "diffuse regular": "REG",
        "localized regular": "REG",
        "diffuse irregular": "IR",
        "localized irregular": "IR",
    },
}


def derm7pt_concept_loaders(preprocess, n_samples, batch_size, num_workers, args):
    from .derma_data import Derm7ptDataset
    from .constants import (
        DERM7_META,
        DERM7_TRAIN_IDX,
        DERM7_VAL_IDX,
        DERM7_TEST_IDX,
        DERM7_FOLDER,
    )

    df = pd.read_csv(DERM7_META)
    train_indexes = list(pd.read_csv(DERM7_TRAIN_IDX)["indexes"])
    val_indexes = list(pd.read_csv(DERM7_VAL_IDX)["indexes"])
    test_indexes = list(pd.read_csv(DERM7_TEST_IDX)["indexes"])
    for COL in df.columns:
        if COL in CONCEPT_TYPE:
            df[COL] = df[COL].map(CONCEPT_TYPE[COL])

    if args.no_concepts == 12:
        df["Atypical Pigment Network"] = df.apply(
            lambda row: {"absent": 0, "typical": 0, "atypical": 1}[
                row["pigment_network"]
            ],
            axis=1,
        )
        df["Typical Pigment Network"] = df.apply(
            lambda row: {"absent": 0, "typical": 1, "atypical": 0}[
                row["pigment_network"]
            ],
            axis=1,
        )

        df["Blue Whitish Veil"] = df.apply(
            lambda row: {"absent": 0, "present": 1}[row["blue_whitish_veil"]], axis=1
        )

        df["Irregular Vascular Structures"] = df.apply(
            lambda row: {"ABS": 0, "REG": 0, "IR": 1}[row["vascular_structures"]],
            axis=1,
        )
        df["Regular Vascular Structures"] = df.apply(
            lambda row: {"ABS": 0, "REG": 1, "IR": 0}[row["vascular_structures"]],
            axis=1,
        )

        df["Irregular Pigmentation"] = df.apply(
            lambda row: {"ABS": 0, "REG": 0, "IR": 1}[row["pigmentation"]], axis=1
        )
        df["Regular Pigmentation"] = df.apply(
            lambda row: {"ABS": 0, "REG": 1, "IR": 0}[row["pigmentation"]], axis=1
        )

        df["Irregular Streaks"] = df.apply(
            lambda row: {"absent": 0, "regular": 0, "irregular": 1}[row["streaks"]],
            axis=1,
        )
        df["Regular Streaks"] = df.apply(
            lambda row: {"absent": 0, "regular": 1, "irregular": 0}[row["streaks"]],
            axis=1,
        )

        df["Regression Structures"] = df.apply(
            lambda row: (1 - int(row["regression_structures"] == "absent")), axis=1
        )

        df["Irregular Dots and Globules"] = df.apply(
            lambda row: {"absent": 0, "regular": 0, "irregular": 1}[
                row["dots_and_globules"]
            ],
            axis=1,
        )
        df["Regular Dots and Globules"] = df.apply(
            lambda row: {"absent": 0, "regular": 1, "irregular": 0}[
                row["dots_and_globules"]
            ],
            axis=1,
        )

        concepts = [
            "Atypical Pigment Network",
            "Typical Pigment Network",
            "Blue Whitish Veil",
            "Irregular Vascular Structures",
            "Regular Vascular Structures",
            "Irregular Pigmentation",
            "Regular Pigmentation",
            "Irregular Streaks",
            "Regular Streaks",
            "Regression Structures",
            "Irregular Dots and Globules",
            "Regular Dots and Globules",
        ]

    # df = df.iloc[train_indexes+val_indexes]
    df_test = df.iloc[test_indexes]
    concept_loaders = {}
    for c_name in concepts:
        pos_df = df[df[c_name] == 1]
        neg_df = df[df[c_name] == 0]
        print(c_name)
        base_dir = os.path.join(DERM7_FOLDER, "images")
        image_key = "derm"

        print(pos_df.shape, neg_df.shape)

        if (pos_df.shape[0] < 2 * n_samples) or (neg_df.shape[0] < 2 * n_samples):
            print("\t Not enough samples! Sampling with replacement for ", c_name)
            pos_df = pos_df.sample(2 * n_samples, replace=True)
            neg_df = neg_df.sample(2 * n_samples, replace=True)
        else:
            pos_df = pos_df.sample(2 * n_samples)
            neg_df = neg_df.sample(2 * n_samples)

        pos_ds = Derm7ptDataset(
            pos_df, base_dir=base_dir, image_key=image_key, transform=preprocess
        )
        neg_ds = Derm7ptDataset(
            neg_df, base_dir=base_dir, image_key=image_key, transform=preprocess
        )
        pos_loader = DataLoader(
            pos_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        neg_loader = DataLoader(
            neg_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        concept_loaders[c_name] = {"pos": pos_loader, "neg": neg_loader}
    return concept_loaders


def get_concept_loaders(args, preprocess, batch_size=100, num_workers=4):
    dataset_name = args.concept_dataset
    n_samples = args.n_samples
    if dataset_name == "derm7pt":
        return derm7pt_concept_loaders(
            preprocess, n_samples, batch_size, num_workers, args
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
