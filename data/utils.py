import random
import torch
import numpy as np


def sample_batch(dataset):

    B = 30
    N = len(dataset)
    frag_ids = list(dataset.keys())
    D = len(dataset[frag_ids[0]][0])

    i_sample = random.sample(frag_ids, 1)[0]
    sims = dataset[i_sample][1]
    frag_ids.remove(i_sample)

    x1 = torch.Tensor(dataset[str(i_sample)][0])
    x2 = np.zeros((B, D))
    y = np.zeros(B)

    for i in range(len(sims)):
        x2[i, :] = dataset[str(sims[i])][0]
        y[i] = 1
        # frag_ids.remove(str(sims[i]))

    for j, random_sample in enumerate(random.sample(frag_ids, B-len(sims))):
        x2[j+len(sims):] = dataset[str(random_sample)][0]

    x2 = torch.Tensor(x2)
    y = torch.Tensor(y)
    input = torch.vstack((x1, x2))

    return input, y