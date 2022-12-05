import torch
import numpy as np

def reproducibility(SEED: int):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)