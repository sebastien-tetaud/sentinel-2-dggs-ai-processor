import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  # or another scheduler

# Adam optimizer with weight decay
optimizer = optim.Adam(
    model.parameters(),
    lr=float(LEARNING_RATE),
    weight_decay=0.0001  # L2 regularization factor
)

# Learning rate scheduler
# Option 1: StepLR (reduces learning rate by gamma every step_size epochs)
scheduler = StepLR(
    optimizer,
    step_size=30,  # decrease LR every 30 epochs
    gamma=0.1      # multiply LR by 0.1 at each step
)

# Option 2: ReduceLROnPlateau (reduces learning rate when a metric plateaus)
"""
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # 'min' for loss, 'max' for accuracy
    factor=0.1,       # multiply LR by this factor
    patience=10,      # number of epochs with no improvement after which LR is reduced
    verbose=True      # print message when LR is reduced
)
"""

# Option 3: CosineAnnealingLR (cosine annealing schedule)
"""
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # maximum number of iterations
    eta_min=1e-6  # minimum learning rate
)
"""


