import torch
import torch.nn as nn


class ClassicModel(nn.Module):
    def __init__(self, config, inpt_shp):
        super(ClassicModel, self).__init__()
        self.depth = config.depth
        self.nodes = config.n_qubits
        self.config = config

        self.pre_net = nn.Linear(inpt_shp, self.nodes)
        for i in range(self.depth):
            setattr(self, f"Linear_{i}", nn.Linear(self.nodes, self.nodes))
            setattr(self, f"ReLU_{i}", nn.ReLU())

        self.output = nn.Linear(self.nodes, 2)

        print(self)

    def forward(self, input_features):
        input_features = self.pre_net(input_features)

        for i in range(self.depth):
            input_features = getattr(self, f"Linear_{i}")(input_features)
            input_features = getattr(self, f"ReLU_{i}")(input_features)

        output = self.output(input_features)

        return output
