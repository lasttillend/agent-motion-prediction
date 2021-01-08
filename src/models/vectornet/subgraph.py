import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions

class SubgraphNetLayer(nn.Module):
    
    def __init__(self, input_channels: int=128, hidden_channels: int=64):
        super().__init__()
        self.fc = nn.Linear(input_channels, hidden_channels)  # single fully connected network
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Input is node features of a single polyline. Each node corresponds to a vector of the polyline.

        Args:
            input (torch.Tensor): size = [num_nodes_in_polyline, num_node_features]
        
        Returns:
            (torch.Tensor): size = [num_nodes_in_polyline, 2 * hidden_channels]
        """
        hidden = self.fc(input).unsqueeze(0)
        encode_data = F.relu(F.layer_norm(hidden, hidden.size()[1:]))  # layer norm and relu activation
        kernel_size = encode_data.size()[1]  # kernel_size = num_nodes_in_polyline
        maxpool = nn.MaxPool1d(kernel_size)  # max pool over all nodes
        polyline_feature = maxpool(encode_data.transpose(1, 2)).squeeze()
        polyline_feature = polyline_feature.repeat(kernel_size, 1)
        output = torch.cat([encode_data.squeeze(), polyline_feature], 1)
        return output

class SubgraphNet(nn.Module):
        
    def __init__(self, input_channels: int):
        super().__init__()
        self.sublayer1 = SubgraphNetLayer(input_channels)
        self.sublayer2 = SubgraphNetLayer()
        self.sublayer3 = SubgraphNetLayer()  # output = 128, which is 2 * hidden_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out1 = self.sublayer1(input)  # output size = [num_nodes_in_polyline, 2 * hidden_channels]_
        out2 = self.sublayer2(out1)   # output size = [num_nodes_in_polyline, 2 * hidden_channels]
        out3 = self.sublayer3(out2)   # output size = [num_nodes_in_polyline, 2 * hidden_channels]
        
        kernel_size = out3.size()[0]  
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(out3.unsqueeze(1).transpose(0, 2)).squeeze()
        return polyline_feature
