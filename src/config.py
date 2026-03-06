from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Instagram JSON Schema ─────────────────────────────────────────────────────

class Dimensions(BaseModel):
    """Post image dimensions."""
    height: int = Field(..., gt=0)
    width: int  = Field(..., gt=0)


class CaptionNode(BaseModel):
    """Leaf node carrying caption text."""
    text: str = Field(default="")


class CaptionEdge(BaseModel):
    """Edge wrapper around a caption node."""
    node: CaptionNode


class EdgeMediaToCaption(BaseModel):
    """Caption edge list for an Instagram post."""
    edges: List[CaptionEdge] = Field(default_factory=list)


class ThumbnailResource(BaseModel):
    """One resolution entry in thumbnail_resources."""
    config_height: int
    config_width: int
    src: str


class Location(BaseModel):
    """Optional location tag on an Instagram post."""
    id: Optional[str]               = None
    name: Optional[str]             = None
    slug: Optional[str]             = None
    has_public_page: Optional[bool] = None


class InstagramPostRecord(BaseModel):
    """Validated schema for one record in pavbhaji.json."""

    id: str
    shortcode: str
    display_url: str
    dimensions: Dimensions
    is_video: bool                            = False
    edge_media_to_caption: EdgeMediaToCaption
    edge_liked_by: Dict[str, Any]             = Field(default_factory=dict)
    edge_media_preview_like: Dict[str, Any]   = Field(default_factory=dict)
    edge_media_to_comment: Dict[str, Any]     = Field(default_factory=dict)
    tags: List[str]                           = Field(default_factory=list)
    taken_at_timestamp: int                   = Field(..., gt=0)
    location: Optional[Location]              = None
    thumbnail_resources: List[ThumbnailResource] = Field(default_factory=list)
    thumbnail_src: Optional[str]              = None
    urls: List[str]                           = Field(default_factory=list)

    @field_validator("display_url")
    @classmethod
    def display_url_must_be_image(cls, v: str) -> str:
        """Ensure display_url ends with a supported image extension."""
        clean = v.split("?")[0].lower()
        if not any(clean.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
            raise ValueError(f"display_url does not appear to be an image URL: {v}")
        return v

    @model_validator(mode="after")
    def normalise_tags(self) -> "InstagramPostRecord":
        """Lowercase and strip leading # from all hashtags."""
        self.tags = [t.lower().strip("#") for t in self.tags]
        return self

    @property
    def filename(self) -> str:
        """Image filename on disk — last URL segment stripped of query params."""
        return self.display_url.split("/")[-1].split("?")[0]

    @property
    def caption_text(self) -> str:
        """Safely return the first caption text, empty string if absent."""
        if self.edge_media_to_caption.edges:
            return self.edge_media_to_caption.edges[0].node.text
        return ""

    @property
    def like_count(self) -> int:
        """Like count with zero fallback."""
        return self.edge_liked_by.get("count", 0)

    @property
    def comment_count(self) -> int:
        """Comment count with zero fallback."""
        return self.edge_media_to_comment.get("count", 0)


# ─── Dataset Config ────────────────────────────────────────────────────────────

class DatasetConfig(BaseModel):
    """Dataset loading, splitting, and sampling configuration."""

    dataset_root: str      = Field(..., description="Path to the dataset/ folder")
    json_path: str         = Field(..., description="Path to pavbhaji.json")
    sample_fraction: float = Field(
        default=1.0, gt=0.0, le=1.0,
        description="Set 0.1 for fast 10% pipeline validation."
    )
    train_split: float     = Field(default=0.70, gt=0.0, lt=1.0)
    val_split: float       = Field(default=0.15, gt=0.0, lt=1.0)
    test_split: float      = Field(default=0.15, gt=0.0, lt=1.0)
    random_seed: int       = Field(default=42)

    @model_validator(mode="after")
    def splits_must_sum_to_one(self) -> "DatasetConfig":
        """Validate that train + val + test splits add up to exactly 1.0."""
        total = round(self.train_split + self.val_split + self.test_split, 6)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Splits must sum to 1.0, got {total:.6f}")
        return self


# ─── Model Config ──────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """Classifier model selection and hyperparameters."""

    model_type: Literal[
        "logistic_regression",
        "lightgbm",
        "xgboost",
        "svm",
        "mbert",
        "indicbert",
        "muril",
    ] = Field(default="lightgbm")

    # TF-IDF settings (for non-transformer models)
    tfidf_max_features: int       = Field(default=50_000, ge=1000)
    tfidf_ngram_min: int          = Field(default=1, ge=1)
    tfidf_ngram_max: int          = Field(default=3, ge=1)
    use_engineered_features: bool = Field(
        default=True,
        description="Stack hand-crafted numeric features alongside TF-IDF vectors"
    )

    # Transformer settings
    pretrained_model_name: str = Field(default="bert-base-multilingual-cased")
    max_seq_length: int        = Field(default=128, ge=16, le=512)
    learning_rate: float       = Field(default=2e-5, gt=0.0)
    epochs: int                = Field(default=5, ge=1, le=50)
    batch_size: int            = Field(default=32, ge=4, le=256)
    checkpoint_dir: str        = Field(default="outputs/models/")

    @field_validator("learning_rate")
    @classmethod
    def lr_in_sensible_range(cls, v: float) -> float:
        """Reject learning rates that are almost certainly misconfigured."""
        if v > 0.1:
            raise ValueError(f"LR {v} is too high. Typical range: 1e-5 to 1e-2.")
        return v

    @model_validator(mode="after")
    def ngram_range_is_valid(self) -> "ModelConfig":
        """Ensure tfidf_ngram_min <= tfidf_ngram_max."""
        if self.tfidf_ngram_min > self.tfidf_ngram_max:
            raise ValueError("tfidf_ngram_min must be <= tfidf_ngram_max")
        return self
