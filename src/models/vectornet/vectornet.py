import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.vectornet.subgraph import SubgraphNet
from models.vectornet.gnn import GraphAttentionNet

class VectorNet(nn.Module):
    
    def __init__(self, cfg, traj_features=6, map_features=5, num_modes=3):
        super().__init__()
        if cfg is None:
            cfg = dict(device=torch.device("cpu"))
        
        self.traj_subgraphnet = SubgraphNet(traj_features)
        self.map_subgraphnet = SubgraphNet(map_features)
        self.graphnet = GraphAttentionNet()
   
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        # decoder
        self.fc = nn.Linear(64, 64)
        nn.init.kaiming_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, self.num_preds + self.num_modes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def _forward_train(self, trajectory_batch, vectormap_batch):
        """
        Args:
            trajectory_batch: [batch_size, num_traj_polylines, num_nodes_on_each_polyline, num_map_attributes]
            vectormap_batch: [batch_size, num_map_polylines, history_num_frames, num_traj_attributes]
        """
        batch_size, num_traj_polylines, _, _ = trajectory_batch.size()
        
        predict_list = []
        for i in range(batch_size):
            polyline_list = []
            vectormap_list = vectormap_batch[i]
            for j in range(num_traj_polylines):
                traj_feature = self.traj_subgraphnet(trajectory_batch[i, j, :, :])
                polyline_list.append(traj_feature.unsqueeze(0))
            for vec_map in vectormap_list:
                map_feature = self.map_subgraphnet(vec_map)
                polyline_list.append(map_feature.unsqueeze(0))
        
            polyline_feature = F.normalize(torch.cat(polyline_list, dim=0), p=2, dim=1)
            out = self.graphnet(polyline_feature)
            decoded_data = self.fc2(F.relu(self.layer_norm(self.fc(out[0].unsqueeze(0)))))
            predict_list.append(decoded_data)
            
        predict_batch = torch.cat(predict_list, dim=0)
        
        pred, confidences = torch.split(predict_batch, self.num_preds, dim=1)
        pred = pred.view(batch_size, self.num_modes, self.future_len, 2)
        assert confidences.shape == (batch_size, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        
        return pred, confidences

    def _forward_test(self, trajectory_batch):
        pass 

    def forward(self, trajectory, vectormap):
        if self.training:
            return self._forward_train(trajectory, vectormap)
        else:
            return self._forward_test(trajectory)
         
