import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from tomo2seg.volume_sequence import VoxelwiseProbabilityGridPotion, VoxelwiseProbabilityGridPotion2

n = 500
n_iter = 50
batch_size = 128

pc = socket.gethostname()
output_file = Path(f"../data/{__file__.split('/')[-1].split('.py')[0]}.pc={pc}.{n=}.{batch_size=}.csv").resolve()

# noinspection DuplicatedCode
random_probas = np.random.rand(n ** 3).reshape(n, n, n)
random_probas /= (random_probas.sum())
random_probas /= (random_probas.sum())

gpg = VoxelwiseProbabilityGridPotion(
    x_range=(0, n), y_range=(0, n), z_range=(0, n),
    probabilities_volume=random_probas.copy(),
    random_state=np.random.RandomState(42),
)

gpg2 = VoxelwiseProbabilityGridPotion2(
    x_range=(0, n), y_range=(0, n), z_range=(0, n),
    probabilities_volume=random_probas.copy(),
    random_state=np.random.RandomState(42),
)

data = {
    "iter": list(range(n_iter)),
    "v1secs": n_iter * [None],
    "v2secs": n_iter * [None],
}

for i in data["iter"]:
    start = time.time()
    gpg.get(batch_size)
    stop = time.time()
    data["v1secs"][i] = stop - start

for i in data["iter"]:
    start = time.time()
    gpg2.get(batch_size)
    stop = time.time()
    data["v2secs"][i] = stop - start

df = pd.DataFrame(data=data).set_index("iter")
df.to_csv(output_file)
df = df.describe().T


if __name__ == "__main__":
    # n = 300
    # n_iter = 30
    # batch_size = 16
    print(tabulate(
        df,
        headers="keys",
        tablefmt="psql",
        floatfmt=".3g",
        numalign="decimal",
        stralign="center",
        showindex=True,
    ))
