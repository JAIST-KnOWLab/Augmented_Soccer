import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os

from retrieval_model import SequenceEncoder

class InferenceRetriever:

    def __init__(
        self, 
        model_path, 
        feature_dim=4096, 
        embed_dim=768, 
        n_heads=8, 
        n_layers=4, 
        dropout=0.1, 
        max_len=5000, 
        device='cuda'
    ):
        """
        Initialize the retrieval engine

        :param model_path: Path to the trained weight file (.pth)

        :param feature_dim: Original video feature dimension (must be consistent with training)

        :param device: Running device ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        

        self.encoder = SequenceEncoder(
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len
        ).to(self.device)
        

        state_dict = torch.load(model_path, map_location=self.device)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_state_dict[k[8:]] = v 
            else:
                new_state_dict[k] = v
                
        self.encoder.load_state_dict(new_state_dict)

            
        self.encoder.eval() 

    def encode_video(self, video_tensor, time_diff=None):
        """
        video_tensor: [Batch, Seq, Feat]
        time_diff: [Batch]
        return: [Batch, Embed]
        """
        with torch.no_grad():
            # [Seq, Feat] -> [1, Seq, Feat]
            if video_tensor.dim() == 2: 
                video_tensor = video_tensor.unsqueeze(0)
                if time_diff is not None:
                    time_diff = time_diff.unsqueeze(0)
            
            video_tensor = video_tensor.to(self.device)
            if time_diff is not None:
                time_diff = time_diff.to(self.device)
            
            embedding = self.encoder(video_tensor, time_diff)
            return embedding.cpu() 

    def retrieve_best_match(self, query_video, candidate_videos, query_time=None, candidate_times=None):

        query_t_tensor = None
        cand_t_tensor = None
        
        if query_time is not None and candidate_times is not None:
            query_t_tensor = torch.tensor([0.0]) 
   
            time_diffs = [query_time - t for t in candidate_times]
            cand_t_tensor = torch.tensor(time_diffs)

        # [1, Embed]
        query_vec = self.encode_video(query_video, query_t_tensor) 
        
    
        candidates_tensor_list = [v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v) for v in candidate_videos]
        # Pad: [Batch, Max_Seq, Feat]
        candidates_batch = pad_sequence(candidates_tensor_list, batch_first=True).to(self.device)
        
        # [Batch, Embed]
        candidate_vecs = self.encode_video(candidates_batch, cand_t_tensor) 
        

        scores = torch.matmul(query_vec.to(self.device), candidate_vecs.to(self.device).T).squeeze(0)
        
        scores = scores.cpu().numpy()
        best_idx = np.argmax(scores)
        
        return best_idx, scores

if __name__ == "__main__":
    print("InferenceRetriever...")
    
