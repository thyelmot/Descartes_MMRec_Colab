import argparse

def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
    parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
    parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=64, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=1, type=int, help='number of gnn layers')
    parser.add_argument('--topk', default=20, type=int, help='K of top K')
    parser.add_argument('--data', default='baby', type=str, help='name of dataset')
    parser.add_argument('--model_type', default='flowmatching_optimized', type=str, help='diffmm / flowmatching_original / flowmatching_optimized')
    parser.add_argument('--ssl_reg', default=1e-2, type=float, help='weight for contrative learning')
    parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
    parser.add_argument("--seed", type=int, default=421, help="random seed")
    parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')
    parser.add_argument('--dims', type=str, default='[1000]')
    parser.add_argument('--d_emb_size', type=int, default=10)
    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--noise_min', type=float, default=0.0001)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--sampling_noise', type=bool, default=False)
    parser.add_argument('--sampling_steps', type=int, default=0)
    parser.add_argument('--rebuild_k', type=int, default=1)
    parser.add_argument('--e_loss', type=float, default=0.1)
    parser.add_argument('--ris_lambda', type=float, default=0.5)
    parser.add_argument('--ris_adj_lambda', type=float, default=0.2)
    parser.add_argument('--trans', type=int, default=0, help='0: R*R, 1: Linear, 2: allrecipes')
    parser.add_argument('--cl_method', type=int, default=0, help='0:m vs m ; 1:m vs main')
    parser.add_argument('--tau_threshold', type=float, default=0.7, help='Doubt threshold for edge pruning')
    parser.add_argument('--alpha_doubt', type=float, default=0.3, help='Weight for internal discrepancy D_item')
    parser.add_argument('--beta_doubt', type=float, default=0.5, help='Weight for preference deviation D_pref')
    parser.add_argument('--gamma_doubt', type=float, default=0.2, help='Weight for popularity bias T_pop')
    parser.add_argument('--epsilon_adv', type=float, default=0.05, help='Perturbation magnitude for FGSM')
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='Adversarial Loss weight')
    parser.add_argument('--adv_warmup_epochs', type=int, default=5, help='Number of epochs before applying Adversarial Training')
    # Colab-specific
    parser.add_argument('--dataset_path', type=str, default='./Datasets', help='Base path to Datasets folder')
    parser.add_argument('--checkpoint_dir', type=str, default='.', help='Directory to save/load checkpoints')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file (optional, useful for Colab)')
    # parse_known_args để tránh conflict với Jupyter/Colab kernel args
    args, _ = parser.parse_known_args()
    return args

args = ParseArgs()


def get_config_from_dict(config_dict):
    """
    Tạo args object từ dict — dùng trong Colab notebook thay cho CLI.
    
    Ví dụ:
        config = {
            'data': 'baby',
            'model_type': 'flowmatching_optimized',
            'epoch': 50,
            ...
        }
        from Params import get_config_from_dict
        args = get_config_from_dict(config)
    """
    # Lấy defaults từ argparse
    base_args = ParseArgs()
    # Override bằng dict
    for key, value in config_dict.items():
        setattr(base_args, key, value)
    return base_args
