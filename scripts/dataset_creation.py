from argparse import ArgumentParser

from src.data_processing.dataset_creator import DatasetCreator

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to generate dataset of avatars with captions.")
    parser.add_argument("--save_path", type=str, help="Path to save generated dataset.")
    parser.add_argument("--n_samples", type=int, help="How many samples to generate.")

    args = parser.parse_args()
    DatasetCreator().create_dataset(**vars(args))
