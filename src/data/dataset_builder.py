import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from src.config import DatasetConfig, InstagramPostRecord
from src.data.feature_engineering import extract_all_features


def build_master_dataframe(
    config: DatasetConfig,
    filename_lookup: Dict[str, InstagramPostRecord],
) -> pd.DataFrame:
    """
    Walk images/0 and images/1 to establish ground-truth labels,
    join each file to its JSON record, extract all features.

    Source of truth for labels = folder name on disk (0 or 1).
    JSON record = primary feature source (may be absent for some files).
    """
    rows     = []
    img_root = Path(config.dataset_root) / "images"

    for folder_name in ("0", "1"):
        folder_path = img_root / folder_name
        if not folder_path.exists():
            print(f"Expected folder missing: {folder_path}")
            continue

        label = int(folder_name)   # 0 = Not Pavbhaji, 1 = Pavbhaji

        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            rec   = filename_lookup.get(fname)   # InstagramPostRecord or None
            feats = extract_all_features(rec)

            rows.append({
                "filename"   : fname,
                "image_path" : str(folder_path / fname),
                "label"      : label,
                "has_json"   : int(rec is not None),
                "shortcode"  : rec.shortcode if rec else None,
                **feats,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset="shortcode", keep="first")
    print(f"\n📊 Dataset Summary")
    print(f"  Total records      : {len(df)}")
    print(f"  Pavbhaji  (1)      : {(df.label == 1).sum()}")
    print(f"  Not Pavbhaji (0)   : {(df.label == 0).sum()}")
    print(f"  With JSON metadata : {df.has_json.sum()}")
    print(f"  Without JSON       : {(df.has_json == 0).sum()}\n")
    return df
