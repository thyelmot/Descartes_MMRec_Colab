# DiffMM + Flow Matching — Google Colab Edition

Triển khai và so sánh 3 mô hình khuyến nghị đa phương thức (Multimodal Recommendation):

| Mô hình | Mô tả |
|---|---|
| **DiffMM** | Diffusion model Markov đa phương thức gốc |
| **Flow Matching Original** | Thay SDE bằng ODE với interpolation thẳng |
| **Flow Matching Optimized** | Vector field liên tục + Euler solver tối ưu hóa |

## 🚀 Chạy trên Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/DiffMM_FlowMatching_Colab/blob/main/DiffMM_Colab.ipynb)

### Bước 1: Chuẩn bị Dataset trên Google Drive

Upload thư mục data lên Drive theo cấu trúc:
```
MyDrive/
└── DiffMM_Data/
    ├── baby/
    │   ├── trnMat.pkl
    │   ├── tstMat.pkl
    │   ├── image_feat.npy   (~221 MB)
    │   └── text_feat.npy    (~28 MB)
    ├── sports/
    │   ├── trnMat.pkl
    │   ├── tstMat.pkl
    │   ├── image_feat.npy   (~574 MB)
    │   └── text_feat.npy    (~72 MB)
    └── tiktok/
        ├── trnMat.pkl
        ├── tstMat.pkl
        ├── image_feat.npy
        ├── text_feat.npy
        └── audio_feat.npy
```

### Bước 2: Mở Notebook

1. Mở [DiffMM_Colab.ipynb](DiffMM_Colab.ipynb) trong Google Colab
2. Vào **Runtime → Change runtime type → GPU**
3. Chạy lần lượt từng cell

### Bước 3: Cấu hình đường dẫn (Cell 3)

```python
DATASET_PATH  = '/content/drive/MyDrive/DiffMM_Data'
CHECKPOINT_DIR = '/content/drive/MyDrive/DiffMM_Checkpoints'
```

### Bước 4: Chọn model và dataset (Cell 4)

```python
config = {
    'data': 'baby',                      # 'baby' | 'sports' | 'tiktok'
    'model_type': 'flowmatching_optimized', # hoặc 'diffmm' / 'flowmatching_original'
    'epoch': 50,
    ...
}
```

## 🔄 Auto-Resume sau khi Colab bị ngắt

Checkpoint được lưu tự động vào Google Drive sau mỗi epoch tốt nhất.
Khi Colab bị ngắt, chỉ cần **chạy lại từ Cell 5** — training sẽ tiếp tục từ điểm bị ngắt.

## 📂 Cấu trúc thư mục

```
DiffMM_FlowMatching_Colab/
├── DiffMM_Colab.ipynb    ← Notebook chính
├── Main.py               ← Training loop + train_colab()
├── Model.py              ← GCN Multimodal model
├── Diffusion.py          ← GaussianDiffusion + FlowMatching_Original
├── FlowMatching.py       ← GraphFlowMatching (CFM Optimized)
├── VelocityModel.py      ← Velocity field network
├── DataHandler.py        ← Data loading (hỗ trợ Drive path)
├── Params.py             ← Hyperparameters (Colab-compatible)
├── Benchmark.py          ← So sánh 3 model × 3 dataset
├── requirements.txt      ← Python dependencies
├── Utils/
│   ├── Utils.py          ← Loss functions
│   └── TimeLogger.py     ← Logging (hỗ trợ Drive log file)
└── Datasets/
    └── README.md         ← Hướng dẫn upload data
```

## ⚙️ Hyperparameter Tốt Nhất

| Tham số | Baby | Sports | TikTok |
|---|---|---|---|
| `reg` | `1e-5` | `1e-5` | `1e-4` |
| `ssl_reg` | `1e-1` | `1e-2` | `1e-2` |
| `keepRate` | `1.0` | `0.5` | `0.5` |
| `e_loss` | `0.01` | `0.01` | `0.1` |
| `trans` | `0` | `0` | `1` |
| `cl_method` | `0` | `0` | `1` |

## 📦 Dependencies

```
torch>=2.0.0 | numpy | scipy | tqdm | pandas
```

## 📖 Tham khảo

- DiffMM: [HKUDS/DiffMM](https://github.com/HKUDS/DiffMM)
- Flow Matching: Lipman et al., "Flow Matching for Generative Modeling" (2022)
