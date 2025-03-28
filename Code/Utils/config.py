import torch.nn as nn
import torch.optim as optim

EMBEDDING = "Gigapath"
MAG = "20x"
LEARNING_RATE = 4e-4
LOSS_FUNCTION = nn.BCEWithLogitsLoss()
OPTIM = "AdamW"
PATIENCE = 7
NUM_EPOCHS = 30

# Warm restart parameters
T_0 = 10  # First restart after 10 epochs
T_mult = 2  # Restart period increases by 2x
eta_min = 1e-6  # Minimum learning rate

MODEL_NAME_TEMPLATE = "Model/model_{}_lr{}_{}.pth"
