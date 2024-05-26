import logging

from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
import os

import numpy as np
import parameters as pr
import torch
from create_dataset import array_trainX, array_trainY, val_dataloader,test_dataloader
from models.knn import WeightedKNearestNeighbors
from utils.helper import get_file_name, metric

weights = 0
knn = WeightedKNearestNeighbors(x=array_trainX,
                                y=array_trainY,
                                k=pr.k,
                                similarity='cosine',
                                weights=weights,
                                learning_rate=10 ** -1,
                                device=pr.device,
                                train_split_ratio=pr.wknn_train_split_ratio)
def report(dataloader):
    pred = []
    logits = []
    targ = []
    for (x, y) in dataloader:
        prediction = knn.predict(x, reduction="score")
        pred.extend(prediction[0])
        logits.extend(prediction[1])
        targ.extend(y.tolist())
    pred = torch.tensor(pred)
    logits = torch.tensor(logits)
    targ = torch.tensor(targ)

    confusion_matrix = np.zeros((2, 2), int)
    for idx, (x, y) in enumerate(zip(logits, targ)):
        confusion_matrix[x, y] += 1
    metric(confusion_matrix, verbose=True)
    print(confusion_matrix)

    limit_n = [5,10,25,100]
    res = []
    for val in limit_n:
        lim_score = np.percentile(pred, 100-val)
        confusion_matrix = np.zeros((2, 2), int)
        for idx, (x, y) in enumerate(zip(logits, targ)):
            if pred[idx] < lim_score:
                continue
            confusion_matrix[x, y] += 1
        print(metric(confusion_matrix))

knn.train(100, 10)
report(val_dataloader)
report(test_dataloader)