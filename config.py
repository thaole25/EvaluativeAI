import argparse

from preprocessing import params


def set_arguments():
    parser = argparse.ArgumentParser(
        description="Skin Cancer WoE + Supervised/Unsupervised"
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        default="concept",
        help="select whether run the concept model or woe model",
        choices=["concept", "woe"],
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ice",
        help="select the algorithm: ice or pcbm",
        choices=["ice", "pcbm"],
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="resnext50",
        metavar="M",
        help="select the CNN backbone model (Resnet50)",
        choices=params.CNN_BACKBONES,
    )
    parser.add_argument(
        "--seed", type=int, default=3, metavar="S", help="random seed (default: 100)"
    )
    parser.add_argument(
        "-nc",
        "--no-concepts",
        type=int,
        default=12,
        metavar="NC",
        help="no of concepts (default: 12)",
    )
    parser.add_argument(
        "--remove-concepts",
        nargs="+",
        default=[],
        help="to remove concepts by concept index",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="save_features threshold parameter",
        default=0.7,
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="mean",
        help="select the image feature by getting mean or max",
        choices=["mean", "max"],
    )
    parser.add_argument(
        "--reducer",
        type=str,
        default="NMF",
        help="select the reducer - choose NA if not use a reducer",
        choices=["NMF", "PCA", "NA"],
    )
    parser.add_argument(
        "--ice-clf",
        type=str,
        default="NA",
        help="select the classifier for the ice model",
    )
    parser.add_argument(
        "--woe-clf",
        type=str,
        default="NA",
        help="select the classifier for the woe model",
    )

    parser.add_argument("--concept-dataset", default="derm7pt", type=str)
    parser.add_argument("--dataset", default="ham10000", type=str)
    parser.add_argument("--out-dir", default="pcbm_output", type=str)
    parser.add_argument(
        "--C",
        nargs="+",
        default=[0.01, 0.1],
        type=float,
        help="Regularization parameter for SVMs.",
    )
    parser.add_argument(
        "--n-samples",
        default=50,
        type=int,
        help="Number of positive/negative samples used to learn concepts.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Sparsity coefficient for elastic net.",
        default=0.99,
    )
    parser.add_argument(
        "--lam", type=float, help="Regularization strength.", default=2e-4
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--pcbm-classifier",
        type=str,
        default="sgd",
        help="the classifier model for pcbm",
    )

    parser.add_argument(
        "-rc",
        "--retrain-concept",
        type=bool,
        default=False,
        metavar="RC",
        help="retrain the concept model (default: False)",
    )
    parser.add_argument(
        "-sc",
        "--save-model",
        type=bool,
        default=False,
        metavar="SC",
        help="save concept model and woe model (default: False)",
    )
    parser.add_argument(
        "-f",
        "--channel-first",
        type=bool,
        default=True,
        metavar="CF",
        help="set channel first (default: True)",
    )
    parser.add_argument(
        "-r",
        "--retrain-model",
        type=bool,
        default=False,
        metavar="R",
        help="retrain the CNN model (default: False)",
    )
    parser.add_argument(
        "-ex",
        "--example",
        type=bool,
        default=False,
        help="generate example images for the study (default: False)",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="debug mode - without adding the result to csv",
    )
    parser.add_argument(
        "-sa",
        "--save-for-app",
        type=bool,
        default=False,
        help="save models for the app",
    )
    parser.add_argument(
        "--train-clf",
        type=bool,
        default=False,
        help="train concepts on a classifier",
    )
    parser.add_argument(
        "--print-instances",
        type=bool,
        default=False,
        help="print instances to select in select_instances.py",
    )
    return parser.parse_args()
