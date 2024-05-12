import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
from torchvision.transforms import v2

import logging
import time
import argparse

from preprocessing import cnn_backbones
import preprocessing.params as params
import keypass

parser = argparse.ArgumentParser(description="EvaSKan")
parser.add_argument("-auth", type=bool, default=False, help="activate authentication")
parser.add_argument(
    "--algo",
    type=str,
    default="ice",
    help="select the algorithm: ice or pcbm",
    choices=["ice", "pcbm"],
)
args = parser.parse_args()

plt.rcParams.update({"font.size": 20})
example_images = [
    "test_data/HAM10000_images_part_1/ISIC_0024880.jpg",  # true label 2
    "test_data/HAM10000_images_part_1/ISIC_0025661.jpg",  # true label 2
    "test_data/HAM10000_images_part_1/ISIC_0026492.jpg",  # true label 0
]
example_hypotheses = ["MEL", "BKL", "BCC"]
DXLABELS = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "resnext50"
SEED = 3
CNN_MODEL_CHECKPOINT = "../save_model/checkpoint-{}-seed{}.pt".format(MODEL, SEED)

if args.algo == "ice":
    NO_CONCEPTS = 8
    CLF = "gnb"
    LAYER_NAME = params.ICE_CONCEPT_LAYER[MODEL]
    DESCRIPTION = """
    # EvaSKan - Evaluative Skin Cancer
    Unsupervised concept learning with Weight of Evidence model (ICE+WOE).

    Please start selecting a dermatoscopic image and your hypothesis to generate the evidence. You can choose one in three examples provided.

    For education and research use only.
    """
    FEATURE_PATH = "Explainers/ICE_HAM_{}_ncomp{}_seed{}_NMF/feature_imgs".format(
        MODEL, NO_CONCEPTS, SEED
    )

else:
    NO_CONCEPTS = 12
    CLF = "gnb"
    LAYER_NAME = params.PCBM_CONCEPT_LAYER[MODEL]
    DESCRIPTION = """
    # EvaSKan - Evaluative Skin Cancer
    Supervised concept learning with Weight of Evidence model (PCBM+WOE).

    Please start selecting a dermatoscopic image and your hypothesis to generate the evidence. You can choose one in three examples provided.

    For education and research use only.
    """
    FEATURE_PATH = "Explainers/PCBM_HAM_{}_ncomp{}_seed{}_0.01_50/feature_imgs".format(
        MODEL, NO_CONCEPTS, SEED
    )


EXP_PATH = "../save_model/{}_Exp_{}_ncomp{}_seed{}_mean_{}.sav".format(
    args.algo.upper(), MODEL, NO_CONCEPTS, SEED, CLF
)
WOE_EXPLAINER = "../save_model/{}_woeexplainer_{}_ncomp{}_seed{}_mean_{}.sav".format(
    args.algo.upper(), MODEL, NO_CONCEPTS, SEED, CLF
)
CONCEPT_MODEL = "../save_model/{}_concept_{}_ncomp{}_seed{}_mean_{}.sav".format(
    args.algo.upper(), MODEL, NO_CONCEPTS, SEED, CLF
)

CACHE_PATH = Path("test_data/cache")
NORMALIZED_NO_AUGMENTED_TRANS = v2.Compose(
    [
        v2.Resize((params.INPUT_RESIZE, params.INPUT_RESIZE)),
        v2.ToTensor(),
        v2.Normalize(params.INPUT_MEAN, params.INPUT_STD),
    ]
)
CONCEPT_NAMES = [
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

Exp = torch.load(EXP_PATH, map_location=torch.device(DEVICE))
woeexplainer = torch.load(WOE_EXPLAINER, map_location=torch.device(DEVICE))
concept_model = torch.load(CONCEPT_MODEL, map_location=torch.device(DEVICE))
cnn_model = cnn_backbones.selected_model(MODEL)
cnn_model.load_state_dict(
    torch.load(CNN_MODEL_CHECKPOINT, map_location=torch.device(DEVICE))[
        "model_state_dict"
    ]
)
cnn_model.to(device=DEVICE)
cnn_model = cnn_model.eval()


def woe_input_image(img_path):
    img = Image.open(img_path).convert("RGB")
    original_x = NORMALIZED_NO_AUGMENTED_TRANS(img).numpy()  # .to(device=params.DEVICE)
    if Exp is None:
        x = concept_model.get_feature(original_x, layer_name=LAYER_NAME)
    else:
        x = Exp.reducer.transform(
            concept_model.get_feature(original_x, layer_name=LAYER_NAME)
        )
    h = x[0]
    x_feature = x.mean(axis=(1, 2))
    x_feature = np.squeeze(x_feature)
    x_feature = torch.tensor(x_feature).to(device=params.DEVICE)
    return original_x, h, x_feature


def plot_woe_and_features(
    Exp, img_x, original_h, woes, woe_feats, color, img_test_path
):
    if len(woe_feats) == 0:
        return plt.figure()  # empty figure

    fig = plt.figure(layout="constrained", figsize=(16, len(woe_feats) * 2))
    subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[1, 2])

    axsLeft = subfigs[0].subplots(1, 1)
    axsLeft.barh(np.arange(len(woes)), woes, align="center", color=color)
    if args.algo == "ice":
        axsLeft.set_yticks(
            np.arange(len(woes)), labels=["Feature " + str(f + 1) for f in woe_feats]
        )
    else:
        axsLeft.set_yticks(
            np.arange(len(woes)), labels=[CONCEPT_NAMES[f] for f in woe_feats]
        )

    rows = len(woe_feats)
    cols = 2
    axsRight = subfigs[1].subplots(rows, cols, sharey=True, width_ratios=[1, 5])
    if axsRight.ndim == 1:
        axsRight = axsRight.reshape(1, -1)
    for i, feat in enumerate(woe_feats[::-1]):
        img_test_feat = img_test_path / ("feature_{}.jpg".format(feat))
        if not os.path.exists(img_test_feat):
            print("regenerate test image feature")
            img_test_feat = Exp.segment_concept_image(
                img_x, original_h, feat, img_test_feat
            )
        img_test = plt.imread(img_test_feat)
        axsRight[i][0].imshow(img_test, interpolation="none")
        axsRight[i][0].get_xaxis().set_ticks([])
        axsRight[i][0].get_yaxis().set_ticks([])
        # axsRight[i][0].set_ylabel(str(feat + 1), rotation=0, labelpad=35)

        img_feat = FEATURE_PATH + "/" + "{}.jpg".format(feat)
        img = plt.imread(img_feat)
        axsRight[i][1].imshow(img, interpolation="none")
        axsRight[i][1].get_xaxis().set_ticks([])
        axsRight[i][1].get_yaxis().set_ticks([])

    axsRight[rows - 1][0].set_xlabel("Test image")
    axsRight[rows - 1][1].set_xlabel("Train images")
    plt.close()
    return fig


def predict(image: str, hypothesis: int) -> list:
    s = time.perf_counter()
    image_id = image.split("/")[-2]
    img_test_path = CACHE_PATH / image_id / args.algo / str(hypothesis)
    if not os.path.exists(img_test_path):
        print("Creating folder ", img_test_path)
        os.makedirs(img_test_path, exist_ok=True)
    original_x, original_h, x_feature = woe_input_image(image)
    _1 = time.perf_counter()
    print("get input: ", _1 - s)
    explain = woeexplainer.explain_for_human(
        x=x_feature, y_pred=hypothesis, units="features", show_bayes=False, plot=False
    )
    _2 = time.perf_counter()
    print("WOE: ", _2 - _1)
    woes = explain.attwoes
    sorted_woeidx = np.argsort(woes)  # from min to max
    neg_woe_feats = []
    neg_woes = []
    pos_woe_feats = []
    pos_woes = []
    for i in sorted_woeidx:
        if woes[i] < 0:
            neg_woe_feats.append(i)
            neg_woes.append(woes[i])
        elif woes[i] > 0:
            pos_woe_feats.append(i)
            pos_woes.append(woes[i])
    _3 = time.perf_counter()
    fig_pos = plot_woe_and_features(
        Exp, original_x, original_h, pos_woes, pos_woe_feats, "skyblue", img_test_path
    )
    _4 = time.perf_counter()
    print("Plot: ", _4 - _3)
    fig_neg = plot_woe_and_features(
        Exp,
        original_x,
        original_h,
        neg_woes[::-1],
        neg_woe_feats[::-1],
        "coral",
        img_test_path,
    )
    _5 = time.perf_counter()
    print("Plot: ", _5 - _4)

    return [fig_pos, fig_neg]


logging.warning("Starting Gradio interface...")
examples = [[i, j] for i, j in zip(example_images, example_hypotheses)]

demo = gr.Blocks(title="EvaSKan")
with demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload a dermatoscopic image", type="filepath"
            )
            hypotheses = gr.Radio(
                DXLABELS,
                label="Your hypothesis",
                info="Please select one hypothesis",
                type="index",
            )
            btn = gr.Button(value="Run")
        with gr.Column():
            examples = gr.Examples(examples=examples, inputs=[input_image, hypotheses])
    with gr.Row():
        with gr.Column():
            output_positive = gr.Plot(label="Evidence For", min_width=300)
        with gr.Column():
            output_negative = gr.Plot(label="Evidence Against", min_width=300)
    btn.click(
        fn=predict,
        inputs=[input_image, hypotheses],
        outputs=[output_positive, output_negative],
    )

if args.auth:
    demo.launch(share=False, auth=(keypass.account, keypass.password))
else:
    demo.launch(share=False)
