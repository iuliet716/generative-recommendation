import argparse
import os

# RecBole 1.2.1 calls torch.load without weights_only=False, which PyTorch >= 2.6 rejects.
# Our checkpoints are generated locally, so restoring the old default is safe.
# Ignored by torch < 2.6.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import pandas as pd

from atomic_id_recommender import recommender
from utils.preprocess import split_items_by_review_counts,write_recbole_atomic_file
from utils.download_data import download_amazon_2023, download_amazon_2014


PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = f"{PROJECT_DIRECTORY}/atomic_id_recommender/config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Atomic-id sequential recommendation baseline")
    parser.add_argument(
        "--model",
        default="SASRec",
        choices=["SASRec", "GRU4Rec", "BERT4Rec"],
        help="RecBole sequential model",
    )
    parser.add_argument(
        "--dataset",
        default="beauty-2023",
        choices=["beauty-2023", "beauty-2014"],
    )
    return parser.parse_args()

def run_beauty_2023(model):
    data_dir = f"{PROJECT_DIRECTORY}/data/Amazon-Reviews-2023/beauty"

    download_amazon_2023(
        data_dir,
        meta_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl",
        review_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl",
    )

    meta = pd.read_csv(f"{data_dir}/meta_raw.csv")
    review = pd.read_csv(f"{data_dir}/review_raw.csv")

    review_counts = split_items_by_review_counts(meta, review)
    meta = meta.merge(review_counts, on='parent_asin')

    inter = review[["user_id", "parent_asin", "rating", "timestamp"]].copy()
    inter = inter.rename(columns={"parent_asin": "item_id"})
    write_recbole_atomic_file(inter, data_dir, 'beauty')

    return recommender.execute(
        model=model,
        dataset_name='beauty',
        data_path=f"{PROJECT_DIRECTORY}/data/Amazon-Reviews-2023",
        config_path=CONFIG_PATH,
        result_path=data_dir,
    )

def run_beauty_2014(model):
    data_dir = f"{PROJECT_DIRECTORY}/data/Amazon-2014/beauty"

    download_amazon_2014(
        data_dir,
        review_url="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
        meta_url="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
    )

    review = pd.read_csv(f"{data_dir}/review_raw.csv")

    inter = review[["reviewerID", "asin", "overall", "unixReviewTime"]].copy()
    inter.columns = ["user_id", "item_id", "rating", "timestamp"]
    write_recbole_atomic_file(inter, data_dir, 'beauty')

    return recommender.execute(
        model=model,
        dataset_name='beauty',
        data_path=f"{PROJECT_DIRECTORY}/data/Amazon-2014",
        config_path=CONFIG_PATH,
        result_path=data_dir,
    )

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "beauty-2023":
        run_beauty_2023(args.model)
    elif args.dataset == "beauty-2014":
        run_beauty_2014(args.model)
