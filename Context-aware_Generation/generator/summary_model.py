import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput
import math

from Qformer import BertConfig, BertLMHeadModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) 
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq, Dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class VisualBartQFormerSummarizer(nn.Module):
    def __init__(
        self,
        feature_dim=4096,      
        num_query_tokens=32,   
        bart_model_name="facebook/bart-large",
        qformer_hidden_layers=2,
        dropout=0.1,
        max_len=500           
    ):
        super().__init__()
        
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        bart_dim = self.bart.config.d_model
        
        self.num_query_tokens = num_query_tokens
        
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = qformer_hidden_layers
        encoder_config.encoder_width = bart_dim 
        
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_tokens
        
        self.video_Qformer = BertLMHeadModel(config=encoder_config)
        
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.feature_proj = nn.Linear(feature_dim, bart_dim)
        self.video_proj = nn.Linear(encoder_config.hidden_size, bart_dim)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, bart_dim),
            nn.ReLU(),
            nn.Linear(bart_dim, bart_dim)
        )
        
        self.ln_vision = nn.LayerNorm(bart_dim)
        self.pos_encoding = PositionalEncoding(bart_dim, dropout, max_len)

    def _prepare_inputs_embeds(self, anchor_raw, hist_raw, time_diff, anchor_mask=None, hist_mask=None):
    
        B = anchor_raw.shape[0]
        
        # Anchor
        anchor_embeds = self.feature_proj(anchor_raw)
        anchor_embeds = self.ln_vision(anchor_embeds)
        anchor_embeds = self.pos_encoding(anchor_embeds)
        
        anchor_queries = self.query_tokens.expand(B, -1, -1)
        
        anchor_output = self.video_Qformer.bert(
            query_embeds=anchor_queries,
            encoder_hidden_states=anchor_embeds,
            encoder_attention_mask=anchor_mask, 
            return_dict=True
        )
        anchor_visual_tokens = self.video_proj(anchor_output.last_hidden_state) 
        
        # History
        hist_embeds = self.feature_proj(hist_raw)
        hist_embeds = self.ln_vision(hist_embeds)
        hist_embeds = self.pos_encoding(hist_embeds)
        
        hist_queries = self.query_tokens.expand(B, -1, -1)
        
        hist_output = self.video_Qformer.bert(
            query_embeds=hist_queries,
            encoder_hidden_states=hist_embeds,
            encoder_attention_mask=hist_mask, 
            return_dict=True
        )
        hist_visual_tokens = self.video_proj(hist_output.last_hidden_state) 
        
        # Time
        t_embed = self.time_mlp(time_diff.unsqueeze(1).float() / 1000.0).unsqueeze(1)
        
        # Concat: [Anchor, History, Time]
        inputs_embeds = torch.cat([anchor_visual_tokens, hist_visual_tokens, t_embed], dim=1)
        return inputs_embeds

    def forward(self, anchor_raw, hist_raw, time_diff, labels=None, anchor_mask=None, hist_mask=None):
        inputs_embeds = self._prepare_inputs_embeds(
            anchor_raw, hist_raw, time_diff, 
            anchor_mask=anchor_mask, hist_mask=hist_mask
        )
        
        encoder = self.bart.get_encoder()
        encoder_outputs = encoder(inputs_embeds=inputs_embeds, return_dict=True)
        
        if labels is not None:
            outputs = self.bart(
                encoder_outputs=encoder_outputs, 
                labels=labels
            )
            return outputs.loss, outputs.logits
        else:
            return None

    def generate(self, anchor_raw, hist_raw, time_diff, max_length=100, anchor_mask=None, hist_mask=None):
        inputs_embeds = self._prepare_inputs_embeds(
            anchor_raw, hist_raw, time_diff, 
            anchor_mask=anchor_mask, hist_mask=hist_mask
        )
        
        encoder = self.bart.get_encoder()
        encoder_outputs = encoder(inputs_embeds=inputs_embeds, return_dict=True)
        
        return self.bart.generate(
            encoder_outputs=encoder_outputs, 
            max_length=max_length, 
            num_beams=4,
            early_stopping=True
        )