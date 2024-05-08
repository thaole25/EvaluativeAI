import torch

import datetime
import csv
import time
import pickle

from preprocessing import params, initdata, cnn_backbones
import config
import utils
from ice import PytorchModelWrapper
from pcbm.concepts import ConceptBank
from pcbm.models import get_model

args = config.set_arguments()
params.set_seed(args.seed)

if __name__ == "__main__":
    stime = time.time()
    now = datetime.datetime.now()
    MODEL_CHECKPOINT = params.MODEL_PATH / "checkpoint-{}-seed{}.pt".format(
        args.model, args.seed
    )
    if args.algo == "ice":
        LAYER_NAME = params.ICE_CONCEPT_LAYER[args.model]
        concept_utils = utils.ConceptUtils(args=args)
    else:
        LAYER_NAME = params.PCBM_CONCEPT_LAYER[args.model]
        print("Loading the concept bank...")
        CONCEPT_BANK = f"pcbm_output/{args.concept_dataset}_{args.dataset}_{args.model}_{args.lr}_{args.n_samples}_{args.seed}.pkl"
        all_concepts = pickle.load(open(CONCEPT_BANK, "rb"))
        all_concept_names = list(all_concepts.keys())
        print(all_concept_names)
        concept_bank = ConceptBank(all_concepts, params.DEVICE)
        args.no_concepts = len(all_concept_names)
        concept_utils = utils.ConceptUtils(args=args, concept_names=all_concept_names)

    print("TIME NOW: ", now)
    print("EXP METHOD: ", args.algo)
    print("CLASSES NAMES:{}".format(params.CLASSES_NAMES))
    print("NO CONCEPTS:{}".format(args.no_concepts))
    print("CNN MODEL: {}".format(args.model))
    print("SEED: ", args.seed)
    print("USE CUTMIX or MIXUP: ", params.USE_MIXUP)
    print("BATCH SIZE: ", params.BATCH_SIZE)
    print("NUM WORKERS: ", params.NUM_WORKERS)
    print("CONCEPT LAYER NAME: {}".format(LAYER_NAME))
    if args.algo == "ice":
        print("REDUCER: {}".format(args.reducer))
        model_row = [now, args.model, args.no_concepts, args.seed]

        backbone_model = cnn_backbones.selected_model(model_name=args.model).to(
            device=params.DEVICE
        )
        backbone_model.load_state_dict(
            torch.load(MODEL_CHECKPOINT, map_location=torch.device(params.DEVICE))[
                "model_state_dict"
            ]
        )
        backbone_model.eval()
    elif args.algo == "pcbm":
        model_row = [
            now,
            args.model,
            args.lr,
            args.n_samples,
            args.seed,
            args.no_concepts,
        ]
        print("CLASSIFIER LAYER: ", args.pcbm_classifier)
        print("LEARNING RATE: ", args.lr)
        print(
            "Number of positive/negative samples used to learn concepts: ",
            args.n_samples,
        )
        backbone_model = get_model(args, backbone_name=f"{args.dataset}_{args.model}")
        backbone_model = backbone_model.to(params.DEVICE)
        backbone_model.eval()

    processed_data = initdata.data_starter(args)

    if args.no_concepts == 2048 and args.run_mode == "woe":
        Exp = None
        concept_model = PytorchModelWrapper(
            backbone_model,
            batch_size=params.BATCH_SIZE,
            predict_target=params.TARGET_CLASSES,
            input_channel_first=True,
            model_channel_first=True,
            input_size=[params.INPUT_CHANNEL, params.INPUT_RESIZE, params.INPUT_RESIZE],
        )
    else:
        if args.algo == "ice":
            Exp, concept_model = concept_utils.get_ice_model(
                backbone_model=backbone_model,
                balanced_train_dl_per_class=processed_data.balanced_per_class,
                original_train_dl_per_class=processed_data.original_per_class,
            )
        else:
            Exp, concept_model = concept_utils.get_pcbm_model(
                concept_bank=concept_bank,
                backbone_model=backbone_model,
                balanced_train_dl_per_class=processed_data.balanced_per_class,
                original_train_dl_per_class=processed_data.original_per_class,
                X_train=processed_data.balanced_X,
                y_train=processed_data.balanced_y,
                X_test=processed_data.X_test,
                y_test=processed_data.y_test,
            )

    if args.run_mode == "concept":
        concept_utils.concept_eval_runner(
            processed_data.X_test,
            processed_data.y_test,
            processed_data.X_test_path,
            Exp,
            concept_model,
            model_row,
        )
        if args.algo == "ice":
            FILE_CONCEPT = params.ICE_RESULT_FILE
        else:
            FILE_CONCEPT = params.PCBM_RESULT_FILE

    elif args.run_mode == "woe":
        utils.woe_runner(
            args,
            processed_data.balanced_X,
            processed_data.balanced_y,
            Exp,
            concept_model,
            processed_data.X_test,
            processed_data.y_test,
            processed_data.X_test_path,
            LAYER_NAME,
            model_row,
        )
        if args.algo == "ice":
            FILE_CONCEPT = params.ICE_WOE_RESULT_FILE
        else:
            FILE_CONCEPT = params.PCBM_WOE_RESULT_FILE

    duration = time.time() - stime
    model_row.append(duration)
    print("Time taken: {}(s), {}(m)".format(duration, duration / 60))
    print("-" * 50)
    if args.run_mode == "woe":
        model_row.append(args.feature_type)
        model_row.append(args.clf)

    if args.algo == "ice":
        model_row.append(args.reducer)
    else:
        model_row.append(args.pcbm_classifier)

    if not args.example and not args.debug:
        with open(FILE_CONCEPT, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(model_row)
