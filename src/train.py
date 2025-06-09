import argparse
import os
import subprocess


def run_script(script_name):
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "train")

    script_path = os.path.join(root_dir, script_name)
    subprocess.run(["python", script_path], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training scripts")
    parser.add_argument(
        "--model",
        choices=["bert", "finetune_fusion", "pretrain_fusion"],
        required=True,
        help="Which model training to run",
    )
    args = parser.parse_args()

    script_map = {
        "bert": "bert_finetune.py",
        "finetune_fusion": "finetune_fusion.py",
        "pretrain_fusion": "train_fusion.py",
    }

    run_script(script_map[args.model])
