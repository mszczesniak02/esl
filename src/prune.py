import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import time
from tqdm import tqdm

import numpy as np

import math

from model import *
from hparams import *
from dataloader import *

from train import train_model


def main() -> int:

    RERUN_LEARNING = False      # might give better results after fine-tuning hparams
    ANALYSIS_DETAILED = False   # shows more data when running validation tests on a model

    print("\n\n Pruning model\n\n")

    model = model_load(path=r"../models/bb.pth")

    accuracy_before, time_before, time_per_sample_before = 0.0, 0.0, 0.0
    accuracy_after, time_after, time_per_sample_after = 0.0, 0.0, 0.0

    if ANALYSIS_DETAILED:
        accuracy_before, time_before, time_per_sample_before = model_analyze(
            model)
    else:
        _, test_dl = dataset_load()
        accuracy_before, time_before, time_per_sample_before = model_measure(
            model, test_dl)

    model = model_prune(model, True)

    if RERUN_LEARNING:
        epochs = 10
        print(f"\n\n Retraining model for {epochs} epochs.\n\n")

        _, criterion, optimizer = model_set(lrate=LEARNING_RATE/10)
        model = train_model(model, criterion, optimizer, epochs,
                            is_pruning=True, print_info=False)

    if ANALYSIS_DETAILED:
        accuracy_after, time_after, time_per_sample_after = model_analyze(
            model)
    else:
        _, test_dl = dataset_load()
        accuracy_after, time_after, time_per_sample_after = model_measure(
            model, test_dl)

    print("\n\nRESULTS:\n\n")
    print("\tBefore:\tAfter:")
    print(f"acc:\t{accuracy_before:.2f}%, {accuracy_after:.2f}% ")
    print(f"time_batch:\t{time_before:.2f} ms, {time_after:.2f} ms ")
    print(
        f"time_sample:\t{time_per_sample_before:.2f} ms, {time_per_sample_after:.2f} ms ")

    # Save pruned model
    model_filename = f"model_pruned_{accuracy_after:.2f}.pth"
    model_save(model, filename=model_filename)
    print(f"Pruned model saved as: {model_filename}")

    return 0


if __name__ == "__main__":

    main()
