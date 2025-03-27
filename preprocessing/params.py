"""Configuration parameters for the skin lesion classification project.

This module contains all the configuration parameters used throughout the project,
including paths, model parameters, data processing settings, and visualization options.
"""

import random
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

# Paths
DATA_PATH = Path("../datasets/HAM10000")
MODEL_PATH = Path("../save_model")
EXP_PATH = Path("Explainers")
EXAMPLE_PATH = Path("Example_Image")
RESULT_PATH = Path("results")
LOGGING_PATH = Path("logging")
CACHE_PATH = Path("test_data/cache")

# Result files
BACKBONE_TRAIN_FILE = RESULT_PATH / "cnn.csv"
ICE_RESULT_FILE = RESULT_PATH / "a_ice_exp.csv"
ICE_WOE_RESULT_FILE = RESULT_PATH / "a_woe_ice_exp.csv"
PCBM_CONCEPT_7PT_FILE = RESULT_PATH / "a_learn_concept_7pt.csv"
PCBM_RESULT_FILE = RESULT_PATH / "a_pcbm_exp.csv"
PCBM_WOE_RESULT_FILE = RESULT_PATH / "a_woe_pcbm_exp.csv"
LOCAL_EVAL_FILE = RESULT_PATH / "a_local_eval.csv"
COUNT_EVIDENCE_FILE = RESULT_PATH / "a_count_evidence.csv"
STATS_CNN_FILE = RESULT_PATH / "a_stats_cnn.csv"
STATS_ICE_FILE = RESULT_PATH / "a_stats_ice.csv"
STATS_ICEWOE_FILE = RESULT_PATH / "a_stats_icewoe.csv"
STATS_7PT_FILE = RESULT_PATH / "a_stats_7pt.csv"
STATS_PCBM_FILE = RESULT_PATH / "a_stats_pcbm.csv"
SELECTED_INSTANCES_FILE = RESULT_PATH / "a_selected_instances.csv"

# Dataset information
LESION_TYPE_DICT: Dict[str, str] = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

LESION_NAMES = list(LESION_TYPE_DICT.keys())
LABEL_FULLNAMES = list(LESION_TYPE_DICT.values())
DXLABELS = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

# Classification mappings
BENIGN_MALIGNANT: Dict[str, str] = {
    "akiec": "precancerous",  # malignant
    "bcc": "malignant",
    "bkl": "benign",
    "df": "benign",
    "mel": "malignant",
    "nv": "benign",
    "vasc": "benign",
}

IDX_TO_TRINARY: Dict[int, str] = {
    0: "precancerous",  # malignant
    1: "malignant",
    2: "benign",
    3: "benign",
    4: "malignant",
    5: "benign",
    6: "benign",
}

IDX_TO_BINARY: Dict[int, str] = {
    0: "malignant",
    1: "malignant",
    2: "benign",
    3: "benign",
    4: "malignant",
    5: "benign",
    6: "benign",
}

# Feature mappings
SKIN_FEATURE_ID_TO_LABEL: Dict[int, str] = {
    0: "Vascular\n Structures",
    1: "Medium Irregular\n Pigmentation",  # 1
    2: "Irregular\n Dots and Globules",
    3: "Whitish Veils",
    4: "Light Irregular\n Pigmentation",  # 2
    5: "Dark Irregular\n Pigmentation",
    6: "Lines",
}

SKIN_PCBM_CONCEPT_NAMES: List[str] = [
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

# Model configuration
NUM_TEST_PER_CLASS = 20
NUM_VAL_PER_CLASS = 20
TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6]
NUM_CLASSES = len(TARGET_CLASSES)
CLASSES_NAMES = ["_".join([str(idx), LESION_NAMES[idx]]) for idx in TARGET_CLASSES]

# Backbone models
CNN_BACKBONES = [
    "resnet50",
    "resnet152",
    "resnext50",
]

# Layer configurations
ICE_CONCEPT_LAYER: Dict[str, str] = {
    "resnet50": "layer4",
    "resnet152": "layer4",
    "resnext50": "layer4",
}

PCBM_CONCEPT_LAYER: Dict[str, str] = {
    "resnet50": "backbone.features.7",
    "resnet152": "backbone.features.7",
    "resnext50": "backbone.features.7",
}

# Training parameters
NUM_EPOCH = 200
EARLY_STOPPING_THRESHOLD = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_WORKERS = 8 if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data processing parameters
INPUT_CHANNEL = 3
INPUT_RESIZE = 224
INPUT_MEAN = [0.76303977, 0.5456458, 0.57004434]  # [0.485, 0.456, 0.406]
INPUT_STD = [0.14092788, 0.1526127, 0.1699702]  # [0.229, 0.224, 0.225]

# Training options
USE_MIXUP = True

# Visualization parameters
FONT_SIZE = 70
DPI = 500

# Computational parameters
CALC_LIMIT = 1e9  # 1e8 3e4
ESTIMATE_NUM = 10

# Weight of Evidence thresholds
WOE_THRESHOLDS: Dict[str, Union[float, np.ndarray]] = {
    "Neutral": 1.15,
    "Substantial": 2.3,
    "Strong": 4.61,
    "Decisive": np.inf,
}

# Example images for evaluation
tutorial_TOTAL_IMGS_DICT = {
    "HAM10000_images_part_2/ISIC_0034117.jpg": 4
}  # correct-low-uncertainty

b_correct_high_un = {
    "HAM10000_images_part_2/ISIC_0033932.jpg": 5,
    "HAM10000_images_part_1/ISIC_0026456.jpg": 6,
    "HAM10000_images_part_1/ISIC_0024647.jpg": 4,
    "HAM10000_images_part_2/ISIC_0031026.jpg": 1,
}

b_correct_low_un = {
    "HAM10000_images_part_1/ISIC_0028438.jpg": 5,
    "HAM10000_images_part_1/ISIC_0026190.jpg": 5,
    "HAM10000_images_part_2/ISIC_0031079.jpg": 5,
    "HAM10000_images_part_1/ISIC_0028510.jpg": 5,
}

b_wrong_high_un = {
    "HAM10000_images_part_2/ISIC_0029665.jpg": 4,
    "HAM10000_images_part_1/ISIC_0025491.jpg": 4,
    "HAM10000_images_part_1/ISIC_0027350.jpg": 5,
    "HAM10000_images_part_2/ISIC_0029783.jpg": 0,
}

b_wrong_low_un = {
    "HAM10000_images_part_1/ISIC_0025554.jpg": 4,
    "HAM10000_images_part_2/ISIC_0031593.jpg": 0,
    "HAM10000_images_part_1/ISIC_0029141.jpg": 2,
    "HAM10000_images_part_1/ISIC_0028583.jpg": 0,
}

# Combine all example images
tutorial_TOTAL_IMGS = list(tutorial_TOTAL_IMGS_DICT.keys())
TOTAL_IMGS_DICT = (
    b_correct_high_un | b_correct_low_un | b_wrong_high_un | b_wrong_low_un
)
TOTAL_IMGS_LIST = list(TOTAL_IMGS_DICT.keys())
TOTAL_IMGS_DICT_INCL_TUTORIAL = TOTAL_IMGS_DICT | tutorial_TOTAL_IMGS_DICT

# Create full labels
FULL_LABELS = [
    "{} ({})".format(full, short)
    for full, short in zip(LABEL_FULLNAMES, DXLABELS)
]

# Evaluation parameters
NUM_IMAGES_PER_CONDITION = 8
CONDITIONS = ["recommendation", "hypothesis"]
DEFAULT_INIT_IMAGE = 0
DEFAULT_INIT_HYPOTHESIS = 0


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True