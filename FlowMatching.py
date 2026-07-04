import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphFlowMatching(nn.Module):
    def __init__(self, sigma_min=1e-4):
        super(GraphFlowMatching, self).__init__()
        self.sigma_min = sigma_min
        
    def compute_path(self, x_0, alpha_0, t):
        r"""
        \psi_t(x_0) = (1 - (1 - \sigma_{min})t)x_0 + t\alpha_0
        """
        if t.dim() < x_0.dim():
            t = t.view(-1, *([1]*(x_0.dim()-1)))
        return (1 - (1 - self.sigma_min) * t) * x_0 + t * alpha_0
    
    def compute_target_velocity(self, x_0, alpha_0):
        r"""
        u_t = \alpha_0 - (1 - \sigma_{min})x_0
        """
        return alpha_0 - (1 - self.sigma_min) * x_0

    def estimate_alpha_0(self, psi_t, v_t, t):
        r"""
        \hat{\alpha}_0 = (1-\sigma_{min})\psi_t + (1 - t(1-\sigma_{min})) v_t
        """
        if t.dim() < psi_t.dim():
            t = t.view(-1, *([1]*(psi_t.dim()-1)))
        sigma = self.sigma_min
        return (1 - sigma) * psi_t + (1 - t * (1 - sigma)) * v_t

    def optimal_transport_pairing(self, x_0, alpha_0):
        """
        [V4] Đã gỡ bỏ Greedy OT để tránh lỗi Mode Collapse (Many-to-One mapping).
        Sử dụng Independent CFM mặc định ghép (x_0[i] -> alpha_0[i]) 
        để đảm bảo sinh đủ đa dạng phân phối dữ liệu (Bijective distribution).
        """
        return alpha_0

    def training_losses(self, model, alpha_0, itmEmbeds, batch_index, model_feats, item_doubt=None, omega=2.0):
        """
        alpha_0: Ground truth interaction matrix (batch_size, num_items)
        """
        batch_size = alpha_0.size(0)
        device = alpha_0.device

        # 1. V4: Adaptive Noise Scheduler (Logit-Normal Sampling)
        # Giúp mô hình tập trung học nhiều hơn ở khoảng giữa của dòng chảy (t gần 0.5)
        # nơi trường vector phức tạp nhất, thay vì chia đều U[0,1]
        z = torch.randn(batch_size, device=device) * 1.2
        t = torch.sigmoid(z)
        
        # 2. Sample x_0 ~ N(0, I)
        x_0 = torch.randn_like(alpha_0)
        
        # --- Descartes V2/V3: Soft-Doubt & Counterfactual Target ---
        if item_doubt is not None:
            # item_doubt có shape (num_items)
            # Expand ra batch_size
            s_d_batch = item_doubt.unsqueeze(0).expand(batch_size, -1)
            
            # Uncertainty-Guided Noise
            noise_scaler = 1.0 + omega * s_d_batch
            x_0 = x_0 * noise_scaler
            
            # Counterfactual Target
            alpha_0 = alpha_0 * (1.0 - s_d_batch)
        # -------------------------------------------------------------
        
        # --- Descartes V4: Independent CFM (No mode collapse) ---
        # alpha_0 = self.optimal_transport_pairing(x_0, alpha_0)
        # ----------------------------

        # 3. Compute path and target velocity
        psi_t = self.compute_path(x_0, alpha_0, t)
        v_target = self.compute_target_velocity(x_0, alpha_0)
        
        # 4. Predict velocity
        v_pred = model(psi_t, t)
        
        # 5. Compute Graph-CFM loss
        mse_cfm = torch.mean((v_pred - v_target) ** 2, dim=list(range(1, len(v_pred.shape))))
        cfm_loss = mse_cfm
        
        # 6. Compute MSI loss
        alpha_hat = self.estimate_alpha_0(psi_t, v_pred, t)
        usr_model_embeds = torch.mm(alpha_hat, model_feats)
        usr_id_embeds = torch.mm(alpha_0, itmEmbeds)
        msi_loss = torch.mean((usr_model_embeds - usr_id_embeds) ** 2, dim=list(range(1, len(usr_model_embeds.shape))))
        
        return cfm_loss, msi_loss

    def euler_solve(self, model, x_start, steps=2): # V3: Chỉ cần 2 steps nhờ OT-CFM
        """
        Solve ODE using Euler method from t=0 to t=1
        V3: Giảm bước giải xuống còn 2 steps (tiết kiệm 60% thời gian)
        """
        device = x_start.device
        batch_size = x_start.size(0)
        
        if steps == 0:
            return x_start

        dt = 1.0 / steps
        x_t = x_start
        
        for i in range(steps):
            t_val = i * dt
            t = torch.full((batch_size,), t_val, device=device)
            v_pred = model(x_t, t)
            x_t = x_t + v_pred * dt
            
        return x_t
