import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json


def sample_batch(dataset, B):

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


class NFM(nn.Module):

    def __init__(self, D):
        super(NFM, self).__init__()
        self.fc1 = nn.Linear(D, D//2)
        self.fc2 = nn.Linear(D//2, D//4)
        self.fc3 = nn.Linear(D//4, D//8)

    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z1 = z[0]
        z2 = z[1:]
        scores = torch.sigmoid(torch.matmul(z1, z2.T))
        return scores
    

class NFMAE(nn.Module):

    def __init__(self, D):
        super(NFMAE, self).__init__()
        self.fc1 = nn.Linear(D, D//2)
        self.fc2 = nn.Linear(D//2, D//4)
        self.fc3 = nn.Linear(D//4, D//8)

        self.fc4 = nn.Linear(D//8, D//4)
        self.fc5 = nn.Linear(D//4, D//2)
        self.fc6 = nn.Linear(D//2, D)

    def forward(self, x):
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        h = F.relu(self.fc3(z))
        z = F.relu(self.fc4(h))
        z = F.relu(self.fc5(z))
        x_out = torch.sigmoid(self.fc6(z))
        return x_out
    
    def feature_vector(self, x):
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        h = F.relu(self.fc3(z))
        return h


#################################### MAIN ####################################

DEVICE = 1
SEED = 47
device = torch.device(f'cuda:{DEVICE}') if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

f = open("notes_dataset.json")
notes_dataset = json.load(f)
f.close()

# model = NFM(1403).to(device)
model = NFMAE(1403).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
bce = torch.nn.BCELoss()

# NFMAE and NFM
for b in range(1000000):
    input, target = sample_batch(notes_dataset, 30)
    input = input.to(device)
    target = target.to(device)
    output = model(input)
    # loss = bce(output, target)
    loss = bce(output, input)
    if b%100==0:
        print(f"Batch {b}, Loss {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "NFMAE.ckpt")

#################################### END OF MAIN ####################################