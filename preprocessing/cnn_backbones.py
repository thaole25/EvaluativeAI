import torch
import torchvision.models as models
import preprocessing.params as params


def set_trainable(m, flag=True):
    for param in m.parameters():
        param.requires_grad = flag
    return m


def selected_model(model_name):
    if model_name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        set_trainable(m)
        num_ftrs = 512
        m.fc = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        set_trainable(m)
        num_ftrs = 2048
        m.fc = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "resnet152":
        m = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        set_trainable(m)
        num_ftrs = 2048
        m.fc = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "resnext50":
        m = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        set_trainable(m)
        num_ftrs = 2048
        m.fc = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "resnext101":
        m = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
        set_trainable(m)
        num_ftrs = 2048
        m.fc = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "resnext101_l":
        m = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.DEFAULT)
        set_trainable(m)
        num_ftrs = 2048
        m.fc = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        set_trainable(m)
        num_ftrs = 1024
        m.classifier = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "densenet201":
        m = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        set_trainable(m)
        num_ftrs = 1920
        m.classifier = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "vgg19":
        m = models.vgg19(weights=models.VGG19_Weights)
        set_trainable(m)
        num_ftrs = 25088
        m.classifier = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    elif model_name == "inception":
        m = models.inception_v3(weights=models.Inception_V3_Weights)
        set_trainable(m)
        num_ftrs = 2048
        m.fc = torch.nn.Linear(num_ftrs, params.NUM_CLASSES)
    else:
        return NameError("Wrong model name")
    return m
