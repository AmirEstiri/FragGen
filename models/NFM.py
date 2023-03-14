import torch
import torch.nn as nn
import torch.nn.functional as F

from data.utils import sample_batch
import json


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


f = open("data/dataset/notes_dataset.json")
notes_dataset = json.load(f)
f.close()

model = NFM(1403)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
bce = torch.nn.BCEWithLogitsLoss()

for b in range(100):
    input, target = sample_batch(notes_dataset)
    output = model(input)
    loss = bce(output, target)
    print(f"Batch {b}, Loss {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()