import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers=2, nhead=4, max_seq_len=100):
        super(FeatureExtractor, self).__init__()
        
        # 1. Gated Linear Unit (GLU) for dynamic feature selection
        self.feature_gate = nn.Linear(num_features, num_features * 2)
        
        # Initialize gate biases to 1.0 so the sigmoid starts "open"
        nn.init.constant_(self.feature_gate.bias[num_features:], 1.0)
        
        self.proj = nn.Linear(num_features, hidden_size)
        self.hidden_norm = nn.LayerNorm(hidden_size) 
        
        # 2. Learnable [CLS] Token & Positional Encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, hidden_size))
        
        self.pos_drop = nn.Dropout(0.1)
        
        # Truncated Normal Initialization
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # [THE FIX] norm_first=True ensures Pre-LN architecture for training stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size * 2, 
            batch_first=True,
            dropout=0.1,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        N, T, C = x.shape
        
        if T > self.pos_embedding.shape[1] - 1:
            raise ValueError(f"Input sequence length ({T}) exceeds max_seq_len buffer.")
            
        # --- GLU Feature Selection ---
        gate_out = self.feature_gate(x)
        linear_val, gate_val = gate_out.chunk(2, dim=-1)
        x_filtered = linear_val * torch.sigmoid(gate_val) 
        
        # Project to hidden size
        h_proj = F.leaky_relu(self.proj(x_filtered), negative_slope=0.1)
        h_proj = self.hidden_norm(h_proj)
        
        # --- [CLS] Token Integration ---
        cls_tokens = self.cls_token.expand(N, -1, -1)
        h_seq = torch.cat((cls_tokens, h_proj), dim=1)
        
        # --- Positional Encoding Integration ---
        h_seq = h_seq + self.pos_embedding[:, :T+1, :]
        h_seq = self.pos_drop(h_seq)
        
        # Pass through Transformer
        h_trans = self.transformer(h_seq)
        
        # Extract the [CLS] token representation
        e = h_trans[:, 0, :] 
        return e