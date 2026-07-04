import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialTrainer(nn.Module):
    """
    Giai đoạn 2: Tự Vấn và Phản Biện (Adversarial Vulnerability Discovery)
    Sử dụng thuật toán FGSM để tìm ra nhiễu cực tiểu đánh lừa mô hình.
    Có cơ chế warmup: Không tấn công trong những epoch đầu để mô hình định hình trước.
    """
    def __init__(self, epsilon=0.05, warmup_epochs=5):
        super(AdversarialTrainer, self).__init__()
        self.epsilon = epsilon
        self.warmup_epochs = warmup_epochs

    def is_active(self, current_epoch):
        return current_epoch >= self.warmup_epochs

    def generate_perturbation(self, model_loss, features):
        """
        Tính toán gradient của loss theo features để tạo FGSM perturbation.
        """
        # Sử dụng torch.autograd.grad để lấy gradient
        # create_graph=False, retain_graph=True vì ta cần gọi loss.backward() sau này cho main optimizer
        try:
            grad = torch.autograd.grad(model_loss, features, retain_graph=True, create_graph=False)[0]
        except RuntimeError:
            # Trong trường hợp features không nằm trong computation graph của model_loss
            return torch.zeros_like(features)
        
        # FGSM / FGM
        perturbation = self.epsilon * F.normalize(grad, p=2, dim=1)
        return perturbation.detach()
