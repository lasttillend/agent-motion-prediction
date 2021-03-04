import torch
import torch.nn as nn
import torch.nn.functional as F


# Because it is a fully-connected graph, there is no need to build a graph
class GraphAttentionNet(nn.Module):
    
    def __init__(self, in_dim=128, key_dim=64, value_dim=64):
        super().__init__()
        self.queryFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.queryFC.weight)

        self.keyFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.keyFC.weight)
        
        self.valueFC = nn.Linear(in_dim, value_dim)
        nn.init.kaiming_normal_(self.valueFC.weight)

        
    def forward(self, polyline_feature):
        """
        Args:
            polyline_feature (torch.tensor): size = [num_polylines, num_polyline_features]
        """
        p_query = F.relu(self.queryFC(polyline_feature))
        p_key = F.relu(self.keyFC(polyline_feature))
        p_value = F.relu(self.valueFC(polyline_feature))
        query_result = p_query.mm(p_key.t())
        query_result = query_result / (p_key.shape[1] ** 0.5)
        attention = F.softmax(query_result, dim=1)
        output = attention.mm(p_value)
        return output + p_query

