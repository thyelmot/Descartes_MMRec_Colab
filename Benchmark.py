import subprocess
import sys
import os
import json
import argparse
import pandas as pd

def run_model(model_type, dataset, epochs, gpu, dataset_path='./Datasets', checkpoint_dir='.'):
    print(f"\n" + "="*60)
    print(f"Training {model_type.upper()} on dataset: {dataset.upper()} ({epochs} epochs)")
    print("="*60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "Main.py")

    cmd = [
        sys.executable, "-u", main_script,
        "--model_type", model_type,
        "--data", dataset,
        "--epoch", str(epochs),
        "--gpu", gpu,
        "--dataset_path", dataset_path,
        "--checkpoint_dir", checkpoint_dir,
    ]

    # Setup params based on original papers for specific datasets
    if dataset == 'tiktok':
        cmd += ["--reg", "1e-4", "--ssl_reg", "1e-2", "--trans", "1", "--e_loss", "0.1", "--cl_method", "1"]
    elif dataset == 'baby':
        cmd += ["--reg", "1e-5", "--ssl_reg", "1e-1", "--keepRate", "1", "--e_loss", "0.01"]
    elif dataset == 'sports':
        cmd += ["--reg", "1e-5", "--ssl_reg", "1e-2", "--keepRate", "0.5", "--e_loss", "0.01"]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        cwd=script_dir
    )

    for line in process.stdout:
        try:
            print(line, end='', flush=True)
        except Exception:
            pass

    process.wait()
    print(f"Finished {model_type.upper()} on {dataset.upper()}! (Exit code: {process.returncode})")

    res_path = os.path.join(checkpoint_dir, f"results_{model_type}_{dataset}.json")
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Result file not found at {res_path}")
        return None


def run_benchmark_from_notebook(datasets=None, models=None, epochs=2, gpu='0',
                                  dataset_path='./Datasets', checkpoint_dir='.'):
    """
    Chạy benchmark từ Colab notebook (không qua CLI).
    Trả về DataFrame kết quả.
    """
    if datasets is None:
        datasets = ['tiktok', 'baby', 'sports']
    if models is None:
        models = ['diffmm', 'flowmatching_original', 'flowmatching_optimized']

    all_results = []

    for dataset in datasets:
        for model in models:
            res = run_model(model, dataset, epochs, gpu, dataset_path, checkpoint_dir)
            if res:
                model_name = {
                    'diffmm': 'DiffMM',
                    'flowmatching_original': 'Flow Matching (Original)',
                    'flowmatching_optimized': 'Flow Matching (Optimized)'
                }.get(model, model)
                all_results.append({
                    'Dataset': dataset.upper(),
                    'Model': model_name,
                    'Recall@20': round(res['recall'], 6),
                    'NDCG@20': round(res['ndcg'], 6),
                    'Precision@20': round(res['precision'], 6),
                    'Best Epoch': res['best_epoch'],
                })

    if all_results:
        df = pd.DataFrame(all_results)
        return df
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='Chạy so sánh toàn diện 3 tập dữ liệu cho 3 mô hình')
    parser.add_argument('--epoch', default=2, type=int, help='Số epoch huấn luyện mặc định là 2 để sinh bảng nhanh')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    parser.add_argument('--dataset_path', default='./Datasets', type=str, help='Đường dẫn tới thư mục Datasets')
    parser.add_argument('--checkpoint_dir', default='.', type=str, help='Thư mục lưu checkpoint và kết quả')
    args_comp, _ = parser.parse_known_args()

    epochs = args_comp.epoch
    gpu = args_comp.gpu
    dataset_path = args_comp.dataset_path
    checkpoint_dir = args_comp.checkpoint_dir

    df = run_benchmark_from_notebook(epochs=epochs, gpu=gpu,
                                     dataset_path=dataset_path,
                                     checkpoint_dir=checkpoint_dir)

    if not df.empty:
        print("\n\n" + "="*80)
        print("FINAL COMPARISON RESULTS ACROSS ALL 3 DATASETS".center(80))
        print("="*80 + "\n")
        print(df.to_string(index=False))

        os.makedirs(checkpoint_dir, exist_ok=True)
        report_path = os.path.join(checkpoint_dir, f"benchmark_report_epoch_{epochs}.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Báo Cáo Benchmark 3 Kiến Trúc Mô Hình\n\n")
            f.write(f"**Số epoch huấn luyện (Demo):** {epochs}\n\n")
            f.write("| Dataset | Model | Recall@20 | NDCG@20 | Precision@20 | Best Epoch |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
            for _, row in df.iterrows():
                f.write(f"| {row['Dataset']} | {row['Model']} | {row['Recall@20']} | {row['NDCG@20']} | {row['Precision@20']} | {row['Best Epoch']} |\n")

            f.write("\n\n## Nhận xét\n")
            f.write("- **DiffMM**: Là mô hình khuyếch tán Markov đa phương thức gốc.\n")
            f.write("- **Flow Matching (Original)**: Là mô hình Flow Matching cơ bản, thay thế SDE bằng ODE.\n")
            f.write("- **Flow Matching (Optimized)**: Sử dụng Vector trường liên tục kết hợp tích phân Euler trực tiếp tối ưu.\n")

        print(f"\nSuccessfully exported full comparison report to '{report_path}'")
    else:
        print("Error: Could not get enough results from models to build the table.")


if __name__ == '__main__':
    main()
