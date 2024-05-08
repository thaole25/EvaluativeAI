import torch

import csv
import datetime
import time

import config
from preprocessing import initdata, params, cnn_backbones
from utils import ConceptUtils, get_metrics
import classifiers


stime = time.time()
args = config.set_arguments()
params.set_seed(args.seed)
now = datetime.datetime.now()
model_row = [now, args.model, args.no_concepts, args.seed]
LAYER_NAME = params.ICE_CONCEPT_LAYER[args.model]
MODEL_CHECKPOINT = params.MODEL_PATH / "checkpoint-{}-seed{}.pt".format(
    args.model, args.seed
)

print("TIME NOW: ", now)
print("EXP METHOD: ICE")
print("CLASSES NAMES:{}".format(params.CLASSES_NAMES))
print("NO CONCEPTS:{}".format(args.no_concepts))
print("CNN MODEL: {}".format(args.model))
print("SEED: ", args.seed)
print("USE CUTMIX or MIXUP: ", params.USE_MIXUP)
print("BATCH SIZE: ", params.BATCH_SIZE)
print("NUM WORKERS: ", params.NUM_WORKERS)
print("CONCEPT LAYER NAME: {}".format(LAYER_NAME))
print("REDUCER: {}".format(args.reducer))
print("FEATURE TYPE: {}".format(args.feature_type))
print("CLF: {}".format(args.clf))


processed_data = initdata.data_starter(args)
backbone_model = cnn_backbones.selected_model(model_name=args.model).to(
    device=params.DEVICE
)
backbone_model.load_state_dict(
    torch.load(MODEL_CHECKPOINT, map_location=torch.device(params.DEVICE))[
        "model_state_dict"
    ]
)
backbone_model.eval()

concept_utils = ConceptUtils(args=args)
Exp, concept_model = concept_utils.get_ice_model(
    backbone_model=backbone_model,
    balanced_train_dl_per_class=processed_data.balanced_per_class,
    original_train_dl_per_class=processed_data.original_per_class,
)

if args.no_concepts == 2048:
    X_features = concept_model.get_feature(
        processed_data.balanced_X, layer_name=LAYER_NAME
    )
    X_test_features = concept_model.get_feature(
        processed_data.X_test, layer_name=LAYER_NAME
    )

else:
    X_features = Exp.reducer.transform(
        concept_model.get_feature(processed_data.balanced_X, layer_name=LAYER_NAME)
    )
    X_test_features = Exp.reducer.transform(
        concept_model.get_feature(processed_data.X_test, layer_name=LAYER_NAME)
    )

if args.feature_type == "mean":
    X_features = X_features.mean(axis=(1, 2))
    X_test_features = X_test_features.mean(axis=(1, 2))
else:
    X_features = X_features.max(axis=(1, 2))
    X_test_features = X_test_features.max(axis=(1, 2))

print(X_features.shape)

classifier = classifiers.factory(model_type=args.clf)
classifier.fit(X_features, processed_data.balanced_y)
train_acc = 100 * classifier.score(X_features, processed_data.balanced_y)
print("Accuracy on train: {:4.2f}%".format(train_acc))
model_row.append(train_acc)

y_test_pred = classifier.predict(X_test_features)
y_preds = y_test_pred.tolist()
acc, binary_acc, sensitivity, specificity, precision, f1_score = get_metrics(
    processed_data.y_test, y_preds
)
print("Accuracy on test: {:4.2f}%".format(acc * 100))
model_row.append(acc * 100)
model_row.append(binary_acc * 100)
model_row.append(sensitivity * 100)
model_row.append(specificity * 100)
model_row.append(precision * 100)
model_row.append(f1_score * 100)
duration = time.time() - stime
model_row.append(duration)
model_row.append(args.feature_type)
model_row.append(args.clf)
model_row.append(args.reducer)

with open(params.ICE_CLFS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(model_row)
