from glob import glob
import numpy as np
import os
import pandas as pd
from collections import Counter, namedtuple
import preprocessing.params as params
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from preprocessing import data_utils


def get_lesion_by_path():
    ham_df = pd.read_csv(params.DATA_PATH / "HAM10000_metadata.csv", sep=",")
    path_lesion_dict = dict(zip(ham_df["image_id"], ham_df["lesion_id"]))
    return path_lesion_dict


def train_test_equal_test_split(X, y, n_per_class, random_state=None):
    sampled = X.groupby(y, sort=False).apply(
        lambda frame: frame.sample(n_per_class)  # random_state=random_state
    )
    mask = sampled.index.get_level_values(1)

    X_train = X.drop(mask)
    X_test = X.loc[mask]
    y_train = y.drop(mask)
    y_test = y.loc[mask]

    return X_train, X_test, y_train, y_test


def dataloader(args):
    imageid_path_dict = {
        os.path.splitext(os.path.basename(x))[0]: x
        for x in glob(os.path.join(params.DATA_PATH, "*", "*.jpg"))
    }
    ham_df = pd.read_csv(params.DATA_PATH / "HAM10000_metadata.csv", sep=",")
    ham_df["path"] = ham_df["image_id"].map(imageid_path_dict.get)
    to_idx = {k: i for i, k in enumerate(params.LESION_TYPE_DICT.keys())}
    ham_df["idx"] = [to_idx[t] for t in ham_df["dx"]]

    y_all = np.array(ham_df["idx"].to_list())
    print("Class Distribution: ", Counter(y_all))

    X, tX, y, ty = train_test_equal_test_split(
        X=ham_df["path"],
        y=ham_df["idx"],
        n_per_class=params.NUM_TEST_PER_CLASS,
        random_state=args.seed,
    )
    X, valX, y, valy = train_test_equal_test_split(
        X, y, n_per_class=params.NUM_VAL_PER_CLASS, random_state=args.seed
    )
    # X, tX, y, ty = train_test_split(ham_df['path'], ham_df['idx'], test_size=0.20, random_state=args.seed, stratify=ham_df["idx"])
    # X, valX, y, valy = train_test_split(X, y, test_size=0.1875, random_state=args.seed, stratify=y)
    # Train-Test-Val = 65-20-15
    X = np.array(X.to_list())
    tX = np.array(tX.to_list())
    y = np.array(y.to_list())
    ty = np.array(ty.to_list())
    valX = np.array(valX.to_list())
    valy = np.array(valy.to_list())

    print(
        "Training data size: X={}, y={}; Validation data size: valX={}, valy={}; Test data size: tX={}, ty={}".format(
            len(X), len(y), len(valX), len(valy), len(tX), len(ty)
        )
    )
    print("Number of train instances per class. Total: ", len(y), Counter(y))
    print("Number of validation instances per class. Total: ", len(valy), Counter(valy))
    print("Number of test instances per class. Total: ", len(ty), Counter(ty))

    # X will return paths, y will return class id
    return X, y, tX, ty, valX, valy


def weighted_random_sampler(args, X, y, augment=True, normalize=True):
    # NOTE: Over-sample ONLY the training data
    class_count = np.unique(y, return_counts=True)[1]
    weight = 1.0 / class_count
    samples_weight = weight[y]
    samples_weight = torch.from_numpy(samples_weight)
    num_sample_each_class = 1000
    num_samples = num_sample_each_class * len(params.TARGET_CLASSES)
    weighted_sampler = WeightedRandomSampler(
        weights=samples_weight,
        num_samples=num_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if not normalize:
        print("Not normalized here...")
        if augment:
            train_dl = DataLoader(
                data_utils.SkinCancerDataset(
                    X, y, data_utils.NO_NORMALIZED_AUGMENTED_TRANS
                ),
                sampler=weighted_sampler,
                batch_size=params.BATCH_SIZE,
                num_workers=params.NUM_WORKERS,
            )
        else:
            train_dl = DataLoader(
                data_utils.SkinCancerDataset(
                    X, y, data_utils.NO_NORMALIZED_NO_AUGMENTED_TRANS
                ),
                sampler=weighted_sampler,
                batch_size=params.BATCH_SIZE,
                num_workers=params.NUM_WORKERS,
            )
    else:
        if augment:
            if args.model == "inception":
                train_dl = DataLoader(
                    data_utils.SkinCancerDataset(
                        X, y, data_utils.INCEPTION_TRAIN_TRANS
                    ),
                    sampler=weighted_sampler,
                    batch_size=params.BATCH_SIZE,
                    num_workers=params.NUM_WORKERS,
                )
            else:
                train_dl = DataLoader(
                    data_utils.SkinCancerDataset(
                        X, y, data_utils.NORMALIZED_AUGMENTED_TRANS
                    ),
                    sampler=weighted_sampler,
                    batch_size=params.BATCH_SIZE,
                    num_workers=params.NUM_WORKERS,
                )
        else:
            if args.model == "inception":
                train_dl = DataLoader(
                    data_utils.SkinCancerDataset(X, y, data_utils.INCEPTION_VAL_TRANS),
                    sampler=weighted_sampler,
                    batch_size=params.BATCH_SIZE,
                    num_workers=params.NUM_WORKERS,
                )
            else:
                train_dl = DataLoader(
                    data_utils.SkinCancerDataset(
                        X, y, data_utils.NORMALIZED_NO_AUGMENTED_TRANS
                    ),
                    sampler=weighted_sampler,
                    batch_size=params.BATCH_SIZE,
                    num_workers=params.NUM_WORKERS,
                )

    return train_dl


def data_starter(args):
    ProcessedData = namedtuple(
        "ProcessedData",
        [
            "original_per_class",
            "balanced_per_class",
            "balanced_X",
            "balanced_y",
            "X_test",
            "y_test",
            "X_test_path",
        ],
    )
    # Get the data
    X_train, y_train, X_test, y_test, X_val, y_val = dataloader(args)
    original_train_dl = DataLoader(
        data_utils.SkinCancerDataset(
            X_train, y_train, data_utils.NORMALIZED_NO_AUGMENTED_TRANS
        ),
        batch_size=params.BATCH_SIZE,
        num_workers=params.NUM_WORKERS,
    )
    weighted_nomarlize_dl = weighted_random_sampler(
        args, X_train, y_train, augment=True, normalize=True
    )

    print("Loader per class for the train data")
    original_train_dl_per_class = data_utils.get_loader_per_class(original_train_dl)
    print("-" * 30)
    weighted_train_dl_per_class = data_utils.get_loader_per_class(weighted_nomarlize_dl)
    weighted_X_train, weighted_y_train, _ = data_utils.loader_to_numpy(
        weighted_nomarlize_dl
    )

    test_dl = DataLoader(
        data_utils.SkinCancerDataset(
            X_test, y_test, data_utils.NORMALIZED_NO_AUGMENTED_TRANS
        ),
        batch_size=params.BATCH_SIZE,
        num_workers=params.NUM_WORKERS,
    )
    X_test, y_test, X_test_paths = data_utils.loader_to_numpy(test_dl)

    return ProcessedData(
        original_train_dl_per_class,
        weighted_train_dl_per_class,
        weighted_X_train,
        weighted_y_train,
        X_test,
        y_test,
        X_test_paths,
    )
