import random
import numpy as np
import os

random.seed(42)
np.random.seed(42)

seeds = [(random.getrandbits(32), i, None) for i in range(160)]

# saving seeds for reproducibility
with open('seeds.txt', 'w') as f:
    for seed in seeds:
        f.write(f"{seed[0]}\n")