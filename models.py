import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from math import gamma as math_gamma
from pytorch_tcn import TCN


class TemporalEncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=1,
                           hidden_size=hidden_dim,
                           num_layers=num_layers,
                           batch_first=True)

    def forward(self, x):
        # Input shape: (batch_size, num_nodes, seq_len)
        batch_size, num_nodes, seq_len = x.shape
        
        
        x = x.reshape(batch_size * num_nodes, seq_len, 1)  # (batch*nodes, seq_len, 1)
        outputs, _ = self.lstm(x)  # (batch*nodes, seq_len, hidden_dim)
        
        # Reshape back to include node dimension
        return outputs.view(batch_size, num_nodes, seq_len, -1)

class TemporalEncoderTCN(nn.Module):
    def __init__(self, input_dim=1, output_size=64, num_channels=[128,128], kernel_size=3):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i-1]
            layers += [
                nn.Conv1d(in_channels, num_channels[i], kernel_size,
                         padding=(kernel_size-1)//2),
                nn.ReLU()
            ]
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Input shape: (batch_size, num_nodes, seq_len)
        batch_size, num_nodes, seq_len = x.shape
        
        # Process each node's time series independently
        x = x.reshape(batch_size * num_nodes, 1, seq_len)  # (batch*nodes, 1, seq_len)
        
        # Temporal processing
        x = self.tcn(x)  # (batch*nodes, channels, seq_len)
        x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)  # (batch*nodes, channels)
        x = self.linear(x)  # (batch*nodes, output_size)
        
        return x.reshape(batch_size, num_nodes, -1)  # (batch, nodes, hidden)

    
class UniMP_FROND(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, 
                 dropout, heads=1):
        super().__init__()
        self.num_layers = num_layers 
        self.input_proj = TransformerConv(input_dim, hidden_dim, heads=heads, concat=False, beta = False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dynamics_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False, beta = False)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.input_proj(x, edge_index)
        x = F.relu(self.input_norm(x))
        x = F.dropout(x, self.dropout, training=self.training)
        for i, (layer, norm) in enumerate(zip(self.dynamics_layers, self.norms)):
            x = F.relu(norm(layer(x, edge_index) + x))
            x = F.dropout(x, self.dropout, training=self.training)
        return x

class SpatioTemporalForecaster(nn.Module):
    def __init__(self, input_dim=1, input_seq_len=7, temporal_num_channels = [128,128], kernel = 3, temporal_hidden=256, gnn_hidden=512,
                 output_dim=5, num_gnn_layers=3, dropout=0.2, heads = 1,
                 temporal_type='tcn'):
        super().__init__()
        self.temporal_type = temporal_type.lower()
        self.output_dim = output_dim
        # Temporal Encoder (now maintains node-level features)
        if self.temporal_type == 'lstm':
            self.temporal_encoder = TemporalEncoderLSTM(
                input_dim=input_dim,  # Each node's time series is univariate
                hidden_dim=temporal_hidden
            )
            self.temporal_pool = nn.Sequential(
            nn.Linear(input_seq_len, 1),  # Learned temporal weighting
            nn.Flatten(-2)  # (batch, nodes, hidden)
            )
        elif self.temporal_type == 'tcn':
            self.temporal_encoder = TemporalEncoderTCN(
                input_dim=input_dim,
                output_size=temporal_hidden,
                num_channels = temporal_num_channels,
                kernel_size = kernel
            )
       
        
        # GNN processes node-level temporal embeddings
        if self.temporal_type == 'tcn' or self.temporal_type == 'lstm':
            self.gnn = UniMP_FROND(
                input_dim=temporal_hidden,
                hidden_dim=gnn_hidden,
                output_dim=gnn_hidden,
                num_layers=num_gnn_layers,
                dropout=dropout,
                heads = heads
            )
        else:
            self.gnn = UniMP_FROND(
                input_dim=input_seq_len,
                hidden_dim=gnn_hidden,
                output_dim=gnn_hidden,
                num_layers=num_gnn_layers,
                dropout=dropout,
                heads = heads
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(gnn_hidden, gnn_hidden*2),
            nn.ReLU(),
            nn.Linear(gnn_hidden*2, output_dim),
        )
        
    def forward(self, x, edge_index):
        # x: (batch_size, num_nodes, seq_len
        
        batch_size, num_nodes = x.shape[:2]
        if self.temporal_type=='tcn':
    
            x_temp = self.temporal_encoder(x)
            x_gnn = x_temp.reshape(batch_size * num_nodes, -1)

        elif self.temporal_type=='lstm':

            x_temp = self.temporal_encoder(x)
            node_features = self.temporal_pool(x_temp.permute(0,1,3,2)).permute(0,1,2)
            x_gnn = node_features.reshape(batch_size * num_nodes, -1)
            
        else:
            x_gnn = x.reshape(batch_size * num_nodes, -1)
        
        # Spatial Processing
        x_gnn_out = self.gnn(x_gnn, edge_index.to(x.device))

        # Decoding (now node-wise)
        x = self.decoder(x_gnn_out)  # (batch*nodes, forecast_horizon)
        
        return x.reshape(batch_size * num_nodes , self.output_dim, 1)