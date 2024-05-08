import datetime
import os
import pickle
import csv

from pcbm.models import get_model
from pcbm.concepts import learn_concept_bank
from pcbm.data import get_concept_loaders
import config
from preprocessing import params, data_utils


def main():
    args = config.set_arguments()
    params.set_seed(args.seed)
    n_samples = args.n_samples
    now = datetime.datetime.now()
    print("TIME NOW: ", now)
    print("BACKBONE MODEL: ", args.model)
    print("SEED: ", args.seed)
    print("LEARNING RATES: ", args.C)
    print(
        "Number of positive/negative samples used to learn concepts: ", args.n_samples
    )
    print("BATCH SIZE: ", params.BATCH_SIZE)
    print("NUM WORKERS: ", params.NUM_WORKERS)
    model_rows = []

    # Bottleneck part of model
    preprocess = data_utils.NORMALIZED_NO_AUGMENTED_TRANS
    backbone = get_model(args, f"{args.dataset}_{args.model}")
    backbone = backbone.to(params.DEVICE)
    backbone = backbone.eval()

    concept_libs = {C: {} for C in args.C}
    # Get the positive and negative loaders for each concept.

    concept_loaders = get_concept_loaders(
        args, preprocess, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS
    )
    no_concepts = len(list(concept_loaders.keys()))

    for concept_name, loaders in concept_loaders.items():
        pos_loader, neg_loader = loaders["pos"], loaders["neg"]
        # Get CAV for each concept using positive/negative image split
        cav_info = learn_concept_bank(
            pos_loader, neg_loader, backbone, n_samples, args.C, device=params.DEVICE
        )

        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        for C in args.C:
            concept_libs[C][concept_name] = cav_info[C]
            print(concept_name, C, cav_info[C][1], cav_info[C][2])
            model_row = [
                now,
                args.model,
                args.n_samples,
                args.seed,
                no_concepts,
                C,
                cav_info[C][1],
                cav_info[C][2],
            ]
            model_rows.append(model_row)

    if not args.debug:
        with open(params.PCBM_CONCEPT_7PT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(model_rows)

        # Save CAV results
        for C in concept_libs.keys():
            lib_path = os.path.join(
                args.out_dir,
                f"{args.concept_dataset}_{args.dataset}_{args.model}_{C}_{args.n_samples}_{args.seed}.pkl",
            )
            with open(lib_path, "wb") as f:
                pickle.dump(concept_libs[C], f)
            print(f"Saved to: {lib_path}")

            total_concepts = len(concept_libs[C].keys())
            print(f"File: {lib_path}, Total: {total_concepts}")


if __name__ == "__main__":
    main()
