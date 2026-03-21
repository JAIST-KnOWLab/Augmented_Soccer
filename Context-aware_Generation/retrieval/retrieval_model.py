import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SequenceEncoder(nn.Module):

    def __init__(
        self, 
        feature_dim,        
        embed_dim,          
        n_heads,            
        n_layers,           
        dropout,            
        max_len,
        use_cls=True       
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.use_cls = use_cls

        self.input_projector = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        if self.use_cls:
            #  A: [CLS] Token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            #  B:  Attention Pooling
            self.attn_layer = nn.Linear(embed_dim, 1)
        
        #  positional_encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim) 
        )
        
    def forward(self, x: torch.Tensor, time_diff: torch.Tensor = None) -> torch.Tensor:
        batch_size = x.shape[0]
        
        #  [B, N, Dim]
        x = self.input_projector(x)
        
        time_embed = None
        if time_diff is not None:
            t = time_diff.unsqueeze(1) / 1000.0 
            time_embed = self.time_mlp(t).unsqueeze(1) # [B, 1, Dim]

        if self.use_cls:
            # ===  A: [CLS]  ===
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            
            if time_embed is not None:
                cls_tokens = cls_tokens + time_embed

            x = torch.cat((cls_tokens, x), dim=1)
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            
            final_vec = x[:, 0]
            
        else:
            # === Attention Pooling ===
            if time_embed is not None:
                x = x + time_embed # [B, N, Dim] + [B, 1, Dim]
                
            x = self.positional_encoding(x)
            x = self.transformer_encoder(x)
            
            # scores: [B, N, 1]
            scores = self.attn_layer(x)
            weights = torch.softmax(scores, dim=1)
            
            final_vec = (x * weights).sum(dim=1)
        
        final_embedding = self.projection_head(final_vec)
        final_embedding = torch.nn.functional.normalize(final_embedding, p=2, dim=1)
        
        return final_embedding

class TripletModel(nn.Module):

    def __init__(
        self, 
        feature_dim, 
        embed_dim, 
        n_heads, 
        n_layers, 
        margin=0.2,     
        dropout=0.1,    
        max_len=5000,
        use_cls=True    
    ):
        super().__init__()
        
        self.encoder = SequenceEncoder(
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len,
            use_cls=use_cls 
        )
        
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative, anchor_t=None, pos_t=None, neg_t=None):
        anchor_vec = self.encoder(anchor, anchor_t)
        positive_vec = self.encoder(positive, pos_t)
        negative_vec = self.encoder(negative, neg_t)
        
        loss = self.triplet_loss(anchor_vec, positive_vec, negative_vec)
        
        return loss, anchor_vec, positive_vec, negative_vec


