import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubtEvaluator(nn.Module):
    """
    Giai đoạn 1 (V3): Mạng Hoài Nghi Tự Động Thích Ứng (Adaptive Doubt Evaluator)
    Sử dụng cơ chế Attention để mô hình tự động học các trọng số alpha, beta, gamma 
    (mức độ quan trọng của Item Discrepancy, Preference Deviation, Popularity) 
    cho TỪNG tương tác (u, i) cụ thể.
    """
    def __init__(self, hidden_dim=64):
        super(DoubtEvaluator, self).__init__()
        # Mạng Attention nội suy trọng số cho 3 thành tố: D_item, D_pref, T_pop
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 3), 
            nn.Softmax(dim=1) # Đảm bảo tổng trọng số alpha + beta + gamma = 1
        )

    def forward(self, adj_matrix_sp, user_emb, item_visual_emb, item_text_emb):
        # Không dùng no_grad() ở đây để attention_net có thể được cập nhật gradient
        
        # 1. Bất đồng thuận nội tại của Item (Visual vs Text)
        c_i = F.cosine_similarity(item_visual_emb, item_text_emb, dim=1)
        D_item = 1.0 - c_i  # (I,)
        
        # 2. Độ phổ biến (Popularity Bias)
        if isinstance(adj_matrix_sp, torch.Tensor) and adj_matrix_sp.is_sparse:
            item_degrees = torch.sparse.sum(adj_matrix_sp, dim=0).to_dense()[:item_visual_emb.shape[0]]
        else:
            item_degrees = torch.tensor(
                np.array(adj_matrix_sp.sum(axis=0)).flatten(), 
                dtype=torch.float32, device=user_emb.device
            )[:item_visual_emb.shape[0]]
            
        T_pop = 1.0 - 1.0 / torch.log(1.0 + item_degrees + 1e-8)  # (I,)
        
        # Lấy danh sách các cạnh (u, i) từ ma trận thưa
        if not isinstance(adj_matrix_sp, torch.Tensor):
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
        
        # 3. Độ lệch pha Sở thích (Preference Deviation)
        item_fusion = (item_visual_emb + item_text_emb) / 2.0
        u_norm = F.normalize(user_emb, p=2, dim=1)
        i_norm = F.normalize(item_fusion, p=2, dim=1)
        
        u_selected = u_norm[rows]  # (nnz, D)
        i_selected = i_norm[cols]  # (nnz, D)
        
        cos_sim = (u_selected * i_selected).sum(dim=1)  # (nnz,)
        D_pref_edges = 1.0 - cos_sim  # (nnz,)
        
        # 4. Adaptive Attention Weights (alpha, beta, gamma)
        edge_features = torch.cat([user_emb[rows], item_fusion[cols]], dim=1) # (nnz, 2D)
        attn_weights = self.attention_net(edge_features) # (nnz, 3)
        
        alpha_dynamic = attn_weights[:, 0]
        beta_dynamic = attn_weights[:, 1]
        gamma_dynamic = attn_weights[:, 2]
        
        # 5. Tính Doubt Score cho mỗi cạnh (Cá nhân hóa)
        S_doubt_edges = (
            alpha_dynamic * D_item[cols] + 
            beta_dynamic * D_pref_edges + 
            gamma_dynamic * T_pop[cols]
        )
        S_doubt_edges = torch.clamp(S_doubt_edges, 0.0, 1.0)
        
        # 6. Per-item doubt (Dùng cho Diffusion)
        # Bắt buộc phải tính từ S_doubt_edges để Gradient chảy về được attention_net
        item_doubt_sum = torch.zeros(item_visual_emb.size(0), device=S_doubt_edges.device)
        item_degree = torch.zeros(item_visual_emb.size(0), device=S_doubt_edges.device)
        
        item_doubt_sum.scatter_add_(0, cols, S_doubt_edges)
        item_degree.scatter_add_(0, cols, torch.ones_like(S_doubt_edges))
        
        # Trung bình cộng hoài nghi của các cạnh nối vào Item
        item_doubt = item_doubt_sum / (item_degree + 1e-8)
        
        # Fallback cho các item cô lập (không có cạnh): dùng trọng số mặc định 0.5
        fallback_doubt = 0.5 * D_item + 0.5 * T_pop
        fallback_doubt = torch.clamp(fallback_doubt, 0.0, 1.0)
        
        item_doubt = torch.where(item_degree > 0, item_doubt, fallback_doubt)
        
        return S_doubt_edges, rows, cols, item_doubt
