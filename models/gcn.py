import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json


class GNFM(nn.Module):

    def __init__(self, D):
        super(GNFM, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(D, D//2),
            nn.ReLU(),
            nn.Linear(D//2, D//4),
            nn.ReLU(),
            nn.Linear(D//4, D//8),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(D//8, D//4),
            nn.ReLU(),
            nn.Linear(D//4, D//2),
            nn.ReLU(),
            nn.Linear(D//2, D),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
    
    def encode(self, x):
        return self.enc(x)
    
    def decode(self, z):
        return self.dec(z)


#################################### MAIN ####################################

DEVICE = 7
SEED = 47
device = torch.device(f'cuda:{DEVICE}') if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

f = open("notes_dataset.json")
notes_dataset = json.load(f)
f.close()

model = GNFM(1403).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
bce = torch.nn.BCELoss()

# GNFM
for e in range(10000):

    z = {}
    for k in notes_dataset.keys():
        x = torch.Tensor(notes_dataset[k][0]).to(device)
        z[k] = model.encode(x)

    z_avg = {}
    for k in z.keys():
        sims_z = [z[k]]
        for k_sim in notes_dataset[k][1]:
            if k_sim in z.keys():
                sims_z.append(z[k_sim])
        z_avg[k] = sum(sims_z)/len(sims_z)

    loss = 0.0
    for k in z_avg.keys():
        x = torch.Tensor(notes_dataset[k][0]).to(device)
        x_recon = model.dec(z_avg[k])
        loss += bce(x_recon, x)
    loss /= len(z_avg)

    if e%1==0:
        print(f"Epoch {e}, Loss {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "GNFM.ckpt")

#################################### END OF MAIN ####################################