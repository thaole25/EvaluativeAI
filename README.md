# Visual Evaluative AI - EvaSKan

**Visual Evaluative AI** is a tool for decision support by providing positive and negative evidence for a given hypothesis. This tool finds high-level human concepts in an image and generates the Weight of Evidence (WoE) for each hypothesis in the decision-making process. We also apply this tool in the skin cancer domain by building a web-based application that allows users to upload a dermatoscopic image, select a hypothesis and analyse their decisions by evaluating the provided evidence.

By applying this tool, we build a web-based application called **EvaSKan** to evaluate dermatoscopic images. Users can select a hypothesis and the application will generate positive/negative evidence for that particular hypothesis.

![demo](img/EvaSKan.png)

## Prerequisites
### Environment
```
Python 3.10.4
CUDA 12.2.0
UCX-CUDA 1.13.1-CUDA-12.2.0
cuDNN 8.9.3.28-CUDA-12.2.0
Graphviz 5.0.0
torch 2.1.2+cu121
torchvision 0.16.2+cu121
```

### Installation
```
virtualenv ~/venvs/venv-3.10.4-cuda12.2 #create the env
source ~/venvs/venv-3.10.4-cuda12.2/bin/activate #activate the env
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

## Code structure
```
datasets
├── 7-point-criteria
└── HAM10000
save_model
EvaluativeAI
├── Explainers
├── README.md
├── app.py
├── classifiers.py
├── config.py
├── eval.py
├── ice
│   ├── __init__.py
│   ├── channel_reducer.py
│   ├── explainer.py
│   ├── model_wrapper.py
│   └── utils.py
├── ice_clfs.py
├── learn_concepts_dataset.py
├── main.py
├── pcbm
│   ├── concepts
│   │   ├── __init__.py
│   │   └── concept_utils.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── concept_loaders.py
│   │   ├── constants.py
│   │   └── derma_data.py
│   └── models
│       ├── __init__.py
│       ├── derma_models.py
│       ├── model_zoo.py
│       └── pcbm_utils.py
├── pcbm_output
├── preprocessing
│   ├── cnn_backbones.py
│   ├── data_utils.py
│   ├── initdata.py
│   └── params.py
├── reproducibility
│   ├── output
│   └── script
│       ├── pretrained.sh
│       └── scratch.sh
├── requirements.txt
├── test_data
├── train_cnn.py
├── utils.py
└── woe
    ├── __init__.py
    ├── explainers.py
    ├── utils.py
    ├── woe.py
    └── woe_utils.py
```

## Datasets
- HAM10000 dataset: Skin lesion classification dataset - [Kaggle Link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Derm7pt: Dermatology concepts dataset - [Link](https://derm.cs.sfu.ca/Welcome.html)

Please put the datasets in the right folder followed the code structure above.

## Usage

### Reproducibility
To reproduce the results in the paper ..., please either use pre-trained models or train from scratch as described below. Then, run `python eval.py` to see the results.

#### Use pre-trained models
- Available pre-trained models
    + Pre-trained CNN backbones Resnet50, ResneXt50 and Resnet152: in the folder `save_model`
    + Pre-trained concept models ICE, PCBM: pickle files in the folder `Explainers`
- Please refer to `reproducibility/script/pretrained.sh` for the training using pre-trained models above

#### Train from scratch
- Step by step to train from scratch
    + Train the CNN backbone model
    + For unsupervised learning concept, train the concept model ICE
    + For supervised learning concept, we first need to train the concept bank using the 7pt checklist dataset, then train the concept model PCBM using the HAM10000 dataset
- Please refer to `reproducibility/script/scratch.sh` for training from scratch

### Run the app
Before running the app, please download the `save_model` folder and `test_data` in `for_app.zip` [here]() and follow the code structure to put them in the right place.

```
# Use unsupervised learning concept model (ICE)
python app.py --algo ice

# Use supervised learning concept model (PCBM)
python app.py --algo pcbm
```

## References
- [WoE package](https://github.com/dmelis/interpretwoe)
- [ICE package](https://github.com/zhangrh93/InvertibleCE)
- [PCBM package](https://github.com/mertyg/post-hoc-cbm)
