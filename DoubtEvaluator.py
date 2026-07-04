import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubtEvaluator(nn.Module):
    """
    Giai đoạn 1: Cơ chế Lọc Hoài nghi (The Doubt Mechanism)
    Tính toán Doubt Score để cắt tỉa các tương tác nhiễu, clickbait.
    Tối ưu hóa bộ nhớ cho các Dataset lớn (như TikTok, Baby).
    """
    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2):
        super(DoubtEvaluator, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_doubt_score(self, adj_matrix, user_emb, item_visual_emb, item_text_emb):
        # Đảm bảo không track gradient trong quá trình pruning để tiết kiệm RAM
        with torch.no_grad():
            # 1. Tính sự bất đồng thuận đa phương thức nội tại của Item (Visual vs Textual)
            c_i = F.cosine_similarity(item_visual_emb, item_text_emb, dim=1)
            D_item = 1.0 - c_i  # (I,)
            
            # 2. Tính sự lệch pha giữa Sở thích User và Đặc trưng Item
            item_fusion = (item_visual_emb + item_text_emb) / 2.0
            
            # TỐI ƯU BỘ NHỚ: Tính Cosine Similarity thông qua ma trận (U, I) thay vì Tensor (U, I, D)
            u_norm = F.normalize(user_emb, p=2, dim=1)
            i_norm = F.normalize(item_fusion, p=2, dim=1)
            A_ui = torch.matmul(u_norm, i_norm.transpose(0, 1)) # (U, I)
            
            D_pref = 1.0 - A_ui # (U, I)
            
            # 3. Tính độ hoài nghi về Popularity Bias
            if isinstance(adj_matrix, torch.Tensor) and adj_matrix.is_sparse:
                # Nếu adj_matrix là sparse tensor của PyTorch
                item_degrees = torch.sparse.sum(adj_matrix, dim=0).to_dense()[:item_visual_emb.shape[0]]
            else:
                # Nếu adj_matrix là SciPy coo_matrix
                import numpy as np
                item_degrees = torch.tensor(np.array(adj_matrix.sum(axis=0)).flatten(), dtype=torch.float32, device=user_emb.device)[:item_visual_emb.shape[0]]
                
            T_pop = 1.0 - 1.0 / torch.log(1.0 + item_degrees + 1e-8) # (I,)
            
            # 4. Tính điểm Doubt Score tổng hợp
            # Reshape D_item và T_pop (I,) -> (1, I) để broadcasting với (U, I)
            D_item_matrix = D_item.unsqueeze(0)
            T_pop_matrix = T_pop.unsqueeze(0)
            
            S_doubt = self.alpha * D_item_matrix + self.beta * D_pref + self.gamma * T_pop_matrix
            return S_doubt

    def forward(self, adj_matrix_sp, user_emb, item_visual_emb, item_text_emb):
        """
        adj_matrix_sp: SciPy coo_matrix or PyTorch sparse tensor shape (U, I)
        Returns: S_doubt tensor shape (U, I)
        """
        with torch.no_grad():
            S_doubt = self.compute_doubt_score(adj_matrix_sp, user_emb, item_visual_emb, item_text_emb)
            
            # Clamp value between 0 and 1
            S_doubt = torch.clamp(S_doubt, 0.0, 1.0)
            return S_doubt

