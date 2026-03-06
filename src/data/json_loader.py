import json
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import ValidationError
from src.config import InstagramPostRecord


def load_and_validate_json(
    json_path: str,
) -> Tuple[List[InstagramPostRecord], List[dict]]:
    """Load pavbhaji.json, validate every record with Pydantic, return (valid, failed)."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: List[dict] = json.load(fh)

    valid:  List[InstagramPostRecord] = []
    failed: List[dict]                = []

    for entry in raw:
        try:
            valid.append(InstagramPostRecord(**entry))
        except ValidationError as exc:
            failed.append({"entry": entry, "error": str(exc)})

    print(f"✅ Valid records  : {len(valid)}")
    print(f"⚠️  Failed records : {len(failed)}")
    return valid, failed


def build_filename_lookup(
    records: List[InstagramPostRecord],
) -> Dict[str, InstagramPostRecord]:
    """Return filename → InstagramPostRecord mapping for O(1) joins with disk files."""
    return {rec.filename: rec for rec in records}
