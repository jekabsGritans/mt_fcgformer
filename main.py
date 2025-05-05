
# import argparse
# import os
# import sys

# import yaml

# from deploy.predict import run_prediction
# from eval.evaluate import run_evaluation
# from train.train import run_training


# def load_config(config_path):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Functional Groups Classification")
#     parser.add_argument("mode", choices=["train", "eval", "predict"], help="Run mode")
#     parser.add_argument("--config", type=str, required=True, help="Path to config file")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     config = load_config(args.config)

#     if args.mode == "train":
#         run_training(config)
#     elif args.mode == "eval":
#         run_evaluation(config)
#     elif args.mode == "predict":
#         run_prediction(config)
#     else:
#         print(f"Unknown mode: {args.mode}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

