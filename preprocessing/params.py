import torch
import numpy as np

from pathlib import Path
import random

DATA_PATH = Path("../datasets/HAM10000")
MODEL_PATH = Path("../save_model")
EXP_PATH = Path("Explainers")
EXAMPLE_PATH = Path("Example_Image")
RESULT_PATH = Path("reproducibility/output")  # Path("results")
BACKBONE_TRAIN_FILE = RESULT_PATH / "cnn.csv"
ICE_RESULT_FILE = RESULT_PATH / "a_ice_exp.csv"
ICE_WOE_RESULT_FILE = RESULT_PATH / "a_woe_ice_exp.csv"
ICE_CLFS_FILE = RESULT_PATH / "a_ice_clfs.csv"
PCBM_CONCEPT_7PT_FILE = RESULT_PATH / "a_learn_concept_7pt.csv"
PCBM_RESULT_FILE = RESULT_PATH / "a_pcbm_exp.csv"
PCBM_WOE_RESULT_FILE = RESULT_PATH / "a_woe_pcbm_exp.csv"
LOCAL_EVAL_FILE = RESULT_PATH / "a_local_eval.csv"
COUNT_EVIDENCE_FILE = RESULT_PATH / "a_count_evidence.csv"
STATS_CNN_FILE = RESULT_PATH / "a_stats_cnn.csv"
STATS_ICE_FILE = RESULT_PATH / "a_stats_ice.csv"
STATS_7PT_FILE = RESULT_PATH / "a_stats_7pt.csv"
STATS_PCBM_FILE = RESULT_PATH / "a_stats_pcbm.csv"

LESION_TYPE_DICT = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesions",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevi",
    "vasc": "vascular lesions",
}
LESION_NAMES = list(LESION_TYPE_DICT.keys())

BENIGN_MALIGNANT = {
    "akiec": "benign",
    "bcc": "malignant",
    "bkl": "benign",
    "df": "benign",
    "mel": "malignant",
    "nv": "benign",
    "vasc": "benign",
}

IDX_TO_BINARY = {
    0: "benign",
    1: "malignant",
    2: "benign",
    3: "benign",
    4: "malignant",
    5: "benign",
    6: "benign",
}

NUM_TEST_PER_CLASS = 20
NUM_VAL_PER_CLASS = 20
TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6]
NUM_CLASSES = len(TARGET_CLASSES)
CLASSES_NAMES = ["_".join([str(idx), LESION_NAMES[idx]]) for idx in TARGET_CLASSES]
CNN_BACKBONES = [
    "resnet50",
    "resnet152",
    "resnext50",
]
ICE_CONCEPT_LAYER = {
    "resnet50": "layer4",
    "resnet152": "layer4",
    "resnext50": "layer4",
}

PCBM_CONCEPT_LAYER = {
    "resnet50": "backbone.features.7",
    "resnet152": "backbone.features.7",
    "resnext50": "backbone.features.7",
}

NUM_EPOCH = 200
EARLY_STOPPING_THRESHOLD = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_WORKERS = 8 if torch.cuda.is_available() else 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_CHANNEL = 3
INPUT_RESIZE = 224
INPUT_MEAN = [0.76303977, 0.5456458, 0.57004434]  # [0.485, 0.456, 0.406]
INPUT_STD = [0.14092788, 0.1526127, 0.1699702]  # [0.229, 0.224, 0.225]

USE_MIXUP = True

FONT_SIZE = 70
DPI = 500
CALC_LIMIT = 1e9  # 1e8 3e4
ESTIMATE_NUM = 10

WOE_THRESHOLDS = {
    "Neutral": 1.15,
    "Substantial": 2.3,
    # 'Strong': 3.35,
    "Strong": 4.61,
    "Decisive": np.inf,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
