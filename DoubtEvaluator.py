import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubtEvaluator(nn.Module):
    """
    Giai đoạn 1: Cơ chế Lọc Hoài nghi (The Doubt Mechanism)
    Tính toán Doubt Score CHỈ cho các tương tác thật sự tồn tại (sparse edges).
    Trả về một vector doubt score cho mỗi tương tác (u, i) trong training set.
    
    V2: Tối ưu bộ nhớ — không tạo ma trận dense (U × I) nữa.
    """
    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2):
        super(DoubtEvaluator, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, adj_matrix_sp, user_emb, item_visual_emb, item_text_emb):
        """
        adj_matrix_sp: SciPy coo_matrix shape (U, I) — training interaction matrix
        user_emb: (U, D) tensor
        item_visual_emb: (I, D) tensor
        item_text_emb: (I, D) tensor
        
        Returns: 
            edge_doubt_scores: dict mapping (u_idx, i_idx) -> doubt_score 
                               for each observed interaction
            item_doubt: (I,) tensor — per-item doubt component (D_item + T_pop)
        """
        with torch.no_grad():
            # 1. Tính sự bất đồng thuận đa phương thức nội tại của Item (Visual vs Textual)
            c_i = F.cosine_similarity(item_visual_emb, item_text_emb, dim=1)
            D_item = 1.0 - c_i  # (I,)
            
            # 2. Tính độ hoài nghi về Popularity Bias
            if isinstance(adj_matrix_sp, torch.Tensor) and adj_matrix_sp.is_sparse:
                item_degrees = torch.sparse.sum(adj_matrix_sp, dim=0).to_dense()[:item_visual_emb.shape[0]]
            else:
                item_degrees = torch.tensor(
                    np.array(adj_matrix_sp.sum(axis=0)).flatten(), 
                    dtype=torch.float32, device=user_emb.device
                )[:item_visual_emb.shape[0]]
                
            T_pop = 1.0 - 1.0 / torch.log(1.0 + item_degrees + 1e-8)  # (I,)
            
            # 3. Tính D_pref CHỈ cho các cạnh tồn tại (sparse-only)
            # Lấy danh sách các cạnh (u, i) từ sparse matrix
            if not isinstance(adj_matrix_sp, torch.Tensor):
                # SciPy coo_matrix
                from scipy.sparse import coo_matrix as coo_type
                if not hasattr(adj_matrix_sp, 'row'):
                    adj_coo = adj_matrix_sp.tocoo()
                else:
                    adj_coo = adj_matrix_sp
                rows = torch.tensor(adj_coo.row, dtype=torch.long, device=user_emb.device)
                cols = torch.tensor(adj_coo.col, dtype=torch.long, device=user_emb.device)
            else:
                indices = adj_matrix_sp._indices()
                rows = indices[0]
                cols = indices[1]
            
            # Tính cosine similarity CHỈ cho các cặp (u, i) tồn tại
            item_fusion = (item_visual_emb + item_text_emb) / 2.0
            u_norm = F.normalize(user_emb, p=2, dim=1)
            i_norm = F.normalize(item_fusion, p=2, dim=1)
            
            # Chỉ lấy embedding của các user-item trong observed edges
            u_selected = u_norm[rows]  # (nnz, D)
            i_selected = i_norm[cols]  # (nnz, D)
            
            # Cosine similarity per-edge
            cos_sim = (u_selected * i_selected).sum(dim=1)  # (nnz,)
            D_pref_edges = 1.0 - cos_sim  # (nnz,)
            
            # 4. Tính Doubt Score cho mỗi cạnh
            S_doubt_edges = (
                self.alpha * D_item[cols] + 
                self.beta * D_pref_edges + 
                self.gamma * T_pop[cols]
            )
            S_doubt_edges = torch.clamp(S_doubt_edges, 0.0, 1.0)
            
            # Per-item doubt (không phụ thuộc user, dùng cho diffusion)
            item_doubt = self.alpha * D_item + self.gamma * T_pop
            item_doubt = torch.clamp(item_doubt, 0.0, 1.0)
            
            return S_doubt_edges, rows, cols, item_doubt
