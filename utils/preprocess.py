import ast
import os

import pandas as pd
from sentence_transformers import SentenceTransformer


HEAD_THRESHOLD = 0.2
TORSO_THRESHOLD = 0.5


def split_items_by_review_counts(meta, review):
    review_counts = review['parent_asin'].value_counts().reindex(meta['parent_asin'], fill_value=0).reset_index()

    review_counts = review['parent_asin'].value_counts().reindex(meta['parent_asin'], fill_value=0).reset_index()
    review_counts = review_counts.sort_values("count", ascending=False).reset_index(drop=True)

    total_interactions = review_counts["count"].sum()
    review_counts["cumulative_ratio"] = review_counts["count"].cumsum() / total_interactions

    review_counts["group"] = "tail"
    review_counts.loc[review_counts["cumulative_ratio"] <= TORSO_THRESHOLD, "group"] = "torso"
    review_counts.loc[review_counts["cumulative_ratio"] <= HEAD_THRESHOLD, "group"] = "head"

    # summary: group count
    head_min_count = review_counts.loc[review_counts["group"] == "head", "count"].min()
    torso_min_count = review_counts.loc[review_counts["group"] == "torso", "count"].min()

    counts = review_counts["group"].value_counts()
    summary = pd.DataFrame({
        "item_count": [counts["head"], counts["torso"], counts["tail"]],
        "review_count": [
            f"reviews > {head_min_count - 1}",
            f"{torso_min_count} ≤ reviews ≤ {head_min_count - 1}",
            f"reviews < {torso_min_count}",
        ],
    }, index=["head", "torso", "tail"])
    print(summary)

    return review_counts

def create_text_feature(row):
    brand = None
    details = ast.literal_eval(row['details'])
    brand_keys = ['Brand', 'Brand Name', 'Manufacturer', 'Sub Brand']
    for key in brand_keys:
      if key in details and details[key] != '':
        brand = str(details[key]).strip()
      brand = str(row['store']).strip() if brand else None
    
    result = {
        'Title': str(row['title']).strip(),
        'Brand': brand,
        'Price': str(row['price']).strip() if row['price'] else None,
        'Skin Type': details.get('Skin Type', None),
        'Hair Type': details.get('Hair Type', None),
        'Scent': details.get('Scent', None),
        'Item Form': details.get('Item Form', None),
        'Color Name': details.get('Color Name', None),
    }
    return ", ".join([f"{k}: {v}" for k, v in result.items() if v is not None])

def write_recbole_atomic_file(inter, output_path, dataset_name):
    """
    Write a 4-column [user, item, rating, timestamp] frame as a RecBole.inter atomic file.
    """
    inter = inter.dropna(subset=[inter.columns[0], inter.columns[1], inter.columns[3]])

    inter = inter.sort_values(inter.columns[3]).drop_duplicates(
        subset=[inter.columns[0], inter.columns[1]], keep="first"
    )

    inter.columns = [
        "user_id:token",
        "item_id:token",
        "rating:float",
        "timestamp:float",
    ]

    output_path = os.path.join(output_path, f"{dataset_name}.inter")
    inter.to_csv(output_path, sep="\t", index=False)

    print(
        f"[atomic] wrote {len(inter):,} interactions "
        f"({inter['user_id:token'].nunique():,} users, "
        f"{inter['item_id:token'].nunique():,} items) -> {output_path}"
    )

def get_items_embeddings(meta):
    sentence_transformer = SentenceTransformer('sentence-transformers/sentence-t5-base')

    meta['text_feature'] = meta.apply(create_text_feature, axis=1)

    item_ids = meta['parent_asin'].tolist()
    item_texts = meta['text_feature'].tolist()

    embeddings = sentence_transformer.encode(item_texts, show_progress_bar=True)

    item_embeddings = {}
    for item_id, item_text in zip(item_ids, embeddings):
        item_embeddings[item_id] = item_text

    meta['item_embeddings'] = meta['parent_asin'].map(item_embeddings)

    return meta
