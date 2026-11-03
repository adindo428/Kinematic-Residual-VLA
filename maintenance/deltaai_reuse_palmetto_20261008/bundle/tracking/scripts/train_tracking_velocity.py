#!/usr/bin/env python3
"""Launch the matched DOM+Velocity run with a deterministic fresh initialization."""
import random
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run
if __name__ == '__main__':
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    import logging
    logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(asctime)s %(message)s')
    run.main()
