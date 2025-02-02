import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

NUM_SAMPLES = 256  # Per second.

# Loads samples and associated labels. Translates time instances into the evenly spaced
# samples.
def load_samples_full(target_path: str, noisy_path: str):
    # t,ax,ay,az,gx,gy,gz,mx,my,mz
    df_noisy = pd.read_csv(noisy_path)
    df_target = pd.read_csv(target_path)
    t0 = df_target.iloc[0, 0]

    data = []
    labels = []
    for (_, row_target), (_, row_noisy) in zip(df_target.iterrows(), df_noisy.iterrows()):
        t = row_target.iloc[0]
        which_second = np.floor(t - t0)
        which_subsample_of_second = ((t - t0) - np.floor(t - t0)) * NUM_SAMPLES
        which_sample = which_second * NUM_SAMPLES + which_subsample_of_second
        while len(data) < which_sample:
            data.append(data[-1])
            labels.append(labels[-1])
        _,ax,ay,az,gx,gy,gz,mx,my,mz = row_noisy
        data.append([ax,ay,az,gx,gy,gz,mx,my,mz])
        _,ax,ay,az,gx,gy,gz,mx,my,mz = row_target
        labels.append([ax,ay,az,gx,gy,gz,mx,my,mz])
    return torch.Tensor(data), torch.Tensor(labels)

def create_batches_from_samples(data):
    datas = []
    for i in range(0, data.shape[0], 40):  # Advance 40 samples between portions of the same set of samples
        j = i + NUM_SAMPLES
        if j <= data.shape[0]:  # Lazy check to make sure we stay in bounds. There's got to be a better way.
            datas.append(data[i:j,:])
    return datas

class SensorDataset(Dataset):
    def __init__(self, sample_identifiers, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for sample_identifier in sample_identifiers:
            data, labels = load_samples_full(sample_identifier + '_noisy.csv', sample_identifier + '_target.csv')
            self.data.extend(create_batches_from_samples(data))
            self.labels.extend(create_batches_from_samples(labels))

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32, requires_grad=True)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

# Neural network should receive 9 channels of sample data over a 1 second
# interval with evenly spaced samples. My unit collected 132 samples per
# second typically. It's tempting to just use 132 samples in the 1 second,
# but since they may vary in the time difference between samples, it makes
# sense to over-sample the dataset and use a bigger number of samples than
# are available in the time period so that the unevenness in timing can be
# modeled and we don't lose readings (unless they are too close together).
# 256 is a little shy of 132*2.
#
# Proposed input shape: (BATCH_SIZE, 256, 9)
# Proposed output shape: (BATCH_SIZE, 256, 9)
#
# During training, use BATCH_SIZE of whatever's clever.
# During inference (evaluation), use BATCH_SIZE of 1.
class NeuralNetwork(nn.Module):
    def __init__(self):
        self.NUM_CHANNELS = 9
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.NUM_CHANNELS * NUM_SAMPLES, self.NUM_CHANNELS * NUM_SAMPLES),
            nn.ReLU(),
            nn.Linear(self.NUM_CHANNELS * NUM_SAMPLES, self.NUM_CHANNELS * NUM_SAMPLES),  # TODO: What's good in the middle?
            nn.ReLU(),
            nn.Linear(self.NUM_CHANNELS * NUM_SAMPLES, self.NUM_CHANNELS * NUM_SAMPLES),  # TODO: Really, we could have fewer items in output since we only care about the tail end of the output (we would have evaluated the model on a portion of the input before the current invocation).
        )

    def forward(self, x):
        x = self.flatten(x)  # (BATCH_SIZE, 256, 9) -> (BATCH_SIZE, 2304)
        x = self.linear_relu_stack(x)
        x = x.unflatten(1, (NUM_SAMPLES, self.NUM_CHANNELS))  # (BATCH_SIZE, 2304) -> (BATCH_SIZE, 256, 9)
        return x


training_items = []
for i in range(0, 800):
    training_items.append(f'data/out_{i:03d}')
test_items = []
for i in range(800, 1000):
    test_items.append(f'data/out_{i:03d}')
training_data = SensorDataset(training_items)
test_data = SensorDataset(test_items)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = NeuralNetwork().to(device)
#for name, param in model.named_parameters():
#    print(f"{name}: requires_grad={param.requires_grad}")

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # Stochastic gradient descent

def train(dataloader, model: NeuralNetwork, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model: NeuralNetwork, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

epochs = 200
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    with torch.no_grad():
        test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
