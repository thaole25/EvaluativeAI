import torch
import torch.nn as nn
import torchvision.models as models

from preprocessing import params


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model(args, backbone_name="ham10000_resnet50", full_model=False):
    if "ham10000" in backbone_name.lower():
        from .derma_models import get_derma_model

        # if args.model == "inception":
        #     model, backbone, model_top = get_derma_model(args, backbone_name.lower())
        MODEL_CHECKPOINT = params.MODEL_PATH / "checkpoint-{}-seed{}.pt".format(
            args.model, args.seed
        )
        NUM_CLASSES = 7
        num_ftrs = 2048
        if args.model == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif args.model == "resnext50":
            model = models.resnext50_32x4d(
                weights=models.ResNeXt50_32X4D_Weights.DEFAULT
            )
        elif args.model == "resnet152":
            model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_CHECKPOINT)["model_state_dict"])
        backbone, _ = ResNetBottom(model), ResNetTop(model)
    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone
    else:
        return backbone
