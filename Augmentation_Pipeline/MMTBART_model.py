import torch
import torch.nn as nn
import pickle as pkl
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import contextlib
import einops
import sys
import io
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.utils.logging import set_verbosity_error
import warnings
from Qformer import BertConfig, BertLMHeadModel

def process_output_tokens(predict_model, tokens):
    output_texts = []
    for output_token in tokens:
        output_text = predict_model.tokenizer.decode(output_token, skip_special_tokens=True).strip()

        output_text = output_text.replace(predict_model.tokenizer.pad_token, "").strip()

        output_texts.append(output_text)
    
    return output_texts
    
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x.to(torch.float32))
    
class Frame_Predict_Event_Model(nn.Module):
    def __init__(self, 
                 lm_ckpt = "facebook/bart-large",
                 tokenizer_ckpt = "facebook/bart-large",
                 max_frame_pos = 200,
                 window = 15,
                 feature_dim = 2048,
                 num_query_tokens = 32,
                 num_video_query_token = 32,
                 device = "cuda:0",
                 inference = False,
                 freeze_bart_embedding=False,
                 **kwargs,
                 ):
        super().__init__()
        if len(kwargs):
            print(f'kwargs not used: {kwargs}')
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_ckpt)
        special_tokens = ["[PLAYER]", "[TEAM]", "[COACH]", "[REFEREE]", "([TEAM])", "[STADIUM]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens}) 
        self.bart = BartForConditionalGeneration.from_pretrained(lm_ckpt)
        self.bart.resize_token_embeddings(len(self.tokenizer))
        self.bos_token_id = self.tokenizer.bos_token_id
        self.ln_vision = LayerNorm(self.bart.config.d_model)
        self.num_query_tokens = num_query_tokens
        self.num_video_query_token = num_video_query_token
        self.inference = inference
        self.feature_proj = nn.Linear(feature_dim, self.bart.config.d_model)
        self.video_proj = nn.Linear(768, self.bart.config.d_model)
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.bart.config.d_model)
        self.window = window


        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,
                                                                             vision_width = self.bart.config.d_model,
                                                                             num_hidden_layers = 2)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # move to device
        self.bart = self.bart.to(self.device)
        self.video_Qformer = self.video_Qformer.to(self.device)


        if freeze_bart_embedding:
            self.bart.model.shared.requires_grad_(False)
            self.bart.model.encoder.embed_tokens.requires_grad_(False)
            for param in self.bart.model.encoder.parameters():
                param.requires_grad = False

        for i, layer in enumerate(self.bart.model.decoder.layers):
            if i < len(self.bart.model.decoder.layers) - 4:
                for param in layer.parameters():
                    param.requires_grad = False

        for name, param in self.bart.model.decoder.named_parameters():
            if "cross_attn" in name:
                param.requires_grad = True


        self.feature_proj = self.feature_proj.to(self.device)
        self.ln_vision = self.ln_vision.to(self.device)
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.video_frame_position_embedding = self.video_frame_position_embedding.to(self.device)


    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers = 2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def forward(self, samples, validating=False):

            set_verbosity_error()
            warnings.filterwarnings("ignore", category=UserWarning)


            video_features = samples['features'].to(self.device)   # [batch_size, window*2, feature_dim]
            input_ids = samples['input_ids'].to(self.device)       # [batch_size, max_pad_len]
            atts_lm = samples['attention_mask'].to(self.device)    # [batch_size, max_pad_len]

            projected_video_features = self.feature_proj(video_features)
            
            try:
                batch_size, time_length, _ = projected_video_features.size()
            except ValueError:
                batch_size, time_length, _, _ = projected_video_features.size()
                
            projected_video_features = self.ln_vision(projected_video_features)
            if len(projected_video_features.size()) != 4:
                projected_video_features = projected_video_features.unsqueeze(-2) # [batch_s, Time_len, 1, bart_hidden_size]

            
            position_ids = torch.arange(time_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids).unsqueeze(-2) 
            
            # Q-former 
            video_embeds = projected_video_features + frame_position_embeddings
            video_embeds = einops.rearrange(video_embeds, 'b t q h -> b (t q) h', b=batch_size, t=time_length)
            
            video_query_tokens = self.video_query_tokens.expand(video_embeds.shape[0], -1, -1).to(video_embeds.device)
            frame_atts = torch.ones(video_embeds.size()[:-1], dtype=torch.long).to(video_embeds.device)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=video_embeds,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )

            video_hidden = self.video_proj(video_query_output.last_hidden_state)

            input_token_embed = self.bart.model.encoder.embed_tokens(input_ids).to(self.device)
            input_embeds = torch.cat((video_hidden, input_token_embed), dim=1).to(self.device) 
            
            mask_prefix = torch.ones(batch_size, self.num_video_query_token, dtype=atts_lm.dtype).to(self.device)
            mask = torch.concat((mask_prefix, atts_lm), dim=1).to(self.device)

            if self.inference:
                return self.generate_text(input_embeds, mask)

            if validating:
                temp_res_text = self.generate_text(input_embeds, mask)
                anonymized = [sublist[-1] for sublist in samples.get("caption_info", [])]
                return temp_res_text, anonymized

                
            targets = samples['labels'].to(self.device)
            
            decoder_input_ids = targets.clone()
            eos_mask = decoder_input_ids == self.tokenizer.eos_token_id
            decoder_input_ids[eos_mask] = self.tokenizer.pad_token_id 
            
            #  Labels -100，Teacher Forcing)
            labels = targets.clone()
            labels = labels[:, 1:]
            pad_fill = torch.full((labels.shape[0], 1), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels, pad_fill], dim=1)  
            labels[labels == self.tokenizer.pad_token_id] = -100

            input_embeds = input_embeds.to(torch.float32)
            
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

            with self.maybe_autocast():
                outputs = self.bart(
                    input_ids=None, 
                    inputs_embeds=input_embeds, 
                    attention_mask=mask,  
                    decoder_input_ids=decoder_input_ids, 
                    labels=labels  
                )
                
            sys.stdout = original_stdout
            
            return outputs.loss

    def generate_text(self, input_embeds, attention_mask):
        input_embeds = input_embeds.to(torch.float32)
        generated_ids = self.bart.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_length=100,
            min_length=30,
            num_beams=8,  
            do_sample=True,  
            top_p=0.9,  
            repetition_penalty=1.2,
            length_penalty=2,
            temperature=0.9,
            decoder_start_token_id=self.tokenizer.bos_token_id
        )
        return process_output_tokens(self, generated_ids)
    
       