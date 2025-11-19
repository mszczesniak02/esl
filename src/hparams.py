import torch
from datetime import datetime

# hyperparameters - tuning resuts visible via tensorboard, check main readme
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 10
LEARNING_RATE = 0.00045040
PATIENCE = 12  # EARLY STOPPING PATIENCE - AMOUNT AFTER WHICH TRAINING STOPS

# global vars for running the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = 4 if DEVICE == "cpu" else 4
MAX_BATCHES = 1250  # for quantization and pruning (running quick validation)

# DATASETS PATH
TEST_DATA_PATH = "../data/fashion-mnist_test.csv"
TRAIN_DATA_PATH = "../data/fashion-mnist_train.csv"

# PATHS FOR SAVING IMAGES
RESULTS_PATH = f"../results/train_{datetime.now().strftime('%m_%d-%H_%M')}/"
FIGURES_PATH = RESULTS_PATH + "assets/"
TENSORBOARD_LOG_DIR = RESULTS_PATH + "fashion_mnist/"
MODELS_DIR = "../models/"  # saved models go here
# model for inference
MODEL_PATH = "../models/model_base.pth"
