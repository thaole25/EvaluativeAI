# Ref: https://www.kaggle.com/code/xinruizhuang/skin-lesion-classification-acc-90-pytorch

import torch
from torch import optim, nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from tqdm import tqdm
import datetime
import csv
import time

from preprocessing import data_utils, cnn_backbones, initdata, params
import config
from utils import get_metrics

stime = time.time()
args = config.set_arguments()
params.set_seed(args.seed)
MODEL_CHECKPOINT = params.MODEL_PATH / "checkpoint-{}-seed{}.pt".format(
    args.model, args.seed
)
TRAINING_LOG = params.MODEL_PATH / "training-log-{}-seed{}.csv".format(
    args.model, args.seed
)
now = datetime.datetime.now()
print("TIME NOW: ", now)
print("MODEL: ", args.model)
print("NUM WORKERS: ", params.NUM_WORKERS)
print("NUM EPOCH: ", params.NUM_EPOCH)
print("LEARNING RATE: ", params.LEARNING_RATE)
print("BATCH SIZE: ", params.BATCH_SIZE)
print("SEED: ", args.seed)
print("USE CUTMIX or MIXUP: ", params.USE_MIXUP)

model_row = [now, args.model, args.seed]
model = cnn_backbones.selected_model(model_name=args.model).to(device=params.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
criterion = nn.CrossEntropyLoss().to(device=params.DEVICE)

if params.USE_MIXUP:
    cutmix = v2.CutMix(num_classes=params.NUM_CLASSES)
    mixup = v2.MixUp(num_classes=params.NUM_CLASSES)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


total_loss_train, total_acc_train = [], []


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        images, labels, _ = data
        N = images.size(0)
        images = images.to(device=params.DEVICE)
        labels = labels.to(device=params.DEVICE)
        if params.USE_MIXUP:
            images, labels = cutmix_or_mixup(images, labels)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        if params.USE_MIXUP:
            train_acc.update(0)
        else:
            train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 100 == 0:
            print(
                "[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]"
                % (epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg)
            )
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    return train_loss.avg, train_acc.avg


def validate(val_loader, model, criterion, optimizer, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels, img_paths = data
            images = images.to(device=params.DEVICE)
            labels = labels.to(device=params.DEVICE)
            N = images.size(0)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
            val_loss.update(criterion(outputs, labels).item())
    print("------------------------------------------------------------")
    print(
        "[epoch %d], [val loss %.5f], [val acc %.5f]"
        % (epoch, val_loss.avg, val_acc.avg)
    )
    print("------------------------------------------------------------")
    return val_loss.avg, val_acc.avg


def run_training(train_loader, val_loader):
    best_val_acc = 0
    best_epoch = 0
    total_loss_val, total_acc_val = ["val_loss"], ["val_acc"]
    total_loss_train, total_acc_train = ["train_loss"], ["train_acc"]
    for epoch in tqdm(range(1, params.NUM_EPOCH + 1)):
        loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)
        loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        total_loss_train.append(loss_train)
        total_acc_train.append(acc_train)
        if acc_val > best_val_acc:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_val": loss_val,
                },
                MODEL_CHECKPOINT,
            )
            best_val_acc = acc_val
            best_epoch = epoch
            print("*****************************************************")
            print(
                "best record: [epoch %d], [val loss %.5f], [val acc %.5f]"
                % (epoch, loss_val, acc_val)
            )
            print("*****************************************************")
        elif epoch - best_epoch >= params.EARLY_STOPPING_THRESHOLD:
            print("Early stopping at epoch %d" % epoch)
            break

    rows = [total_loss_val, total_loss_train, total_acc_val, total_acc_train]
    with open(TRAINING_LOG, "w") as f:
        write = csv.writer(f)
        write.writerows(rows)
        f.close()


X_train, y_train, X_test, y_test, X_val, y_val = initdata.dataloader(args)
train_dl = initdata.weighted_random_sampler(
    args, X_train, y_train, augment=True, normalize=True
)
val_dl = DataLoader(
    data_utils.SkinCancerDataset(
        X_val, y_val, data_utils.NORMALIZED_NO_AUGMENTED_TRANS
    ),
    batch_size=params.BATCH_SIZE,
    num_workers=params.NUM_WORKERS,
)
test_dl = DataLoader(
    data_utils.SkinCancerDataset(
        X_test, y_test, data_utils.NORMALIZED_NO_AUGMENTED_TRANS
    ),
    batch_size=params.BATCH_SIZE,
    num_workers=params.NUM_WORKERS,
)

if args.retrain_model:
    run_training(train_loader=train_dl, val_loader=val_dl)

model.load_state_dict(torch.load(MODEL_CHECKPOINT)["model_state_dict"])
model.eval()
y_preds = []
y_trues = []
with torch.no_grad():
    for i, data in enumerate(test_dl):
        images, labels, img_paths = data
        images = images.to(device=params.DEVICE)
        labels = labels.to(device=params.DEVICE)
        N = images.size(0)
        outputs = model(images)
        prediction = outputs.max(1, keepdim=True)[1]
        y_trues.extend(labels.cpu().numpy())
        y_preds.extend(np.squeeze(prediction.cpu().numpy().T))

acc, binary_acc, sensitivity, specificity, precision, f1_score = get_metrics(
    y_trues, y_preds
)
print(
    "Accuracy of {}, seed {} on test dataset: {}%".format(
        args.model, args.seed, acc * 100
    )
)
model_row.append(acc * 100)
model_row.append(binary_acc * 100)
model_row.append(sensitivity * 100)
model_row.append(specificity * 100)
model_row.append(precision * 100)
model_row.append(f1_score * 100)
cm = confusion_matrix(y_trues, y_preds, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
img_name = params.RESULT_PATH / "{}_seed{}_matrix_test.png".format(
    args.model, args.seed
)
# disp.plot().figure_.savefig(img_name)
duration = time.time() - stime
print("Time taken: {}(s), {}(m)".format(duration, duration / 60))
print("-" * 50)
model_row.append(duration)
with open(params.BACKBONE_TRAIN_FILE, "a", newline="") as cnn_acc_f:
    writer = csv.writer(cnn_acc_f)
    writer.writerow(model_row)
