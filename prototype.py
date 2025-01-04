import torch
from torch import nn

# Neural network should receive 9 channels of sample data over a 1 second
# interval with evenly spaced samples. My unit collected 132 samples per
# second typically. It's tempting to just use 132 samples in the 1 second,
# but since they may vary in the time difference between samples, it makes
# sense to over-sample the dataset and use a bigger number of samples than
# are available in the time period so that the unevenness in timing can be
# modeled and we don't lose readings (unless they are too close together).
# 256 is a little shy of 132*2.
#
# Proposed input shape: (BATCH_SIZE, 9, 256)
# Proposed output shape: (BATCH_SIZE, 9, 256)
#
# During training, use BATCH_SIZE of whatever's clever.
# During inference (evaluation), use BATCH_SIZE of 1.
class NeuralNetwork(nn.Module):
    def __init__(self):
        self.NUM_CHANNELS = 9
        self.NUM_SAMPLES = 256
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.NUM_CHANNELS * self.NUM_SAMPLES, self.NUM_CHANNELS * self.NUM_SAMPLES),
            nn.ReLU(),
            nn.Linear(self.NUM_CHANNELS * self.NUM_SAMPLES, self.NUM_CHANNELS * self.NUM_SAMPLES),  # TODO: What's good in the middle?
            nn.ReLU(),
            nn.Linear(self.NUM_CHANNELS * self.NUM_SAMPLES, self.NUM_CHANNELS * self.NUM_SAMPLES),  # TODO: Really, we could have fewer items in output since we only care about the tail end of the output (we would have evaluated the model on a portion of the input before the current invocation).
        )

    def forward(self, x):
        x = self.flatten(x)  # (BATCH_SIZE, 9, 256) -> (BATCH_SIZE, 2304)
        x = self.linear_relu_stack(x)
        x = x.unflatten(1, (self.NUM_CHANNELS, self.NUM_SAMPLES))  # (BATCH_SIZE, 2304) -> (BATCH_SIZE, 9, 256)
        return x
