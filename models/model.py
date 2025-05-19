import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, d_model = 512, seq_len = 477):
        super().__init__()

        self.type_classifier = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )


    def forward(self, x):
        output = self.type_classifier(x)
        return output
