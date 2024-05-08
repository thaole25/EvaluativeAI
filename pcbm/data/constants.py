import os

# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = "../datasets/7-point-criteria/"  # "/path/to/derm7pt/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_TEST_IDX = os.path.join(DERM7_FOLDER, "meta", "test_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
HAM10K_DATA_DIR = "../datasets/HAM10000/"
