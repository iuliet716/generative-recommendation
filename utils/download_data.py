import ast
import gzip
import json
import os
import urllib.request

from datasets import load_dataset
import pandas as pd

def download_amazon_2023(file_directory, meta_url, review_url):
    meta_dataset = load_dataset(
        "json", 
        data_files=meta_url,
        split='train',
    )

    review_dataset = load_dataset(
        "json",
        data_files=review_url,
        split='train',
    )

    meta = pd.DataFrame(meta_dataset)
    review = pd.DataFrame(review_dataset)

    os.makedirs(file_directory, exist_ok=True)
    meta.to_csv(f"{file_directory}/meta_raw.csv", index=False)
    review.to_csv(f"{file_directory}/review_raw.csv", index=False)

def download_amazon_2014(file_directory, review_url, meta_url):
    os.makedirs(file_directory, exist_ok=True)

    review_gz = f"{file_directory}/reviews_5.json.gz"
    meta_gz = f"{file_directory}/meta.json.gz"
    review_csv = f"{file_directory}/review_raw.csv"
    meta_csv = f"{file_directory}/meta_raw.csv"

    if os.path.exists(review_csv) and os.path.exists(meta_csv):
        print(f"[2014] cached CSVs found in {file_directory}, skipping download")
        return

    for url, dst in [(review_url, review_gz), (meta_url, meta_gz)]:
        if os.path.exists(dst):
            print(f"[2014] cached {os.path.basename(dst)}, skipping download")
            continue
        print(f"[2014] downloading {url}")
        urllib.request.urlretrieve(url, dst)

    review_rows = [
        (r["reviewerID"], r["asin"], r.get("overall"), r.get("unixReviewTime"))
        for r in _parse_gz(review_gz)
    ]
    review = pd.DataFrame(
        review_rows, columns=["reviewerID", "asin", "overall", "unixReviewTime"]
    )
    review.to_csv(review_csv, index=False)

    meta = pd.DataFrame(_parse_gz(meta_gz))
    meta.to_csv(meta_csv, index=False)

    print(
        f"[2014] parsed {len(review):,} reviews, {len(meta):,} items -> {file_directory}"
    )

def _parse_gz(path):
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line.decode("utf-8"))