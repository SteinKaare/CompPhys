import sys
import numpy as np

label = sys.argv[1]

with open(f"test_{label}", "ab") as f:
        np.save(f, 5)