# core/ir/schema.py
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator

Domain = Literal["cv", "nlp", "ml", "other"]

class MetricSpec(BaseModel):
    key: str = Field(..., description="e.g., accuracy, top1, f1")
    target: float = Field(..., ge=0.0, le=1.0, description="Expected metric in [0,1] if normalized; allow paper-scale")
    tolerance_pct: float = Field(2.0, ge=0.0, le=100.0)

class LayerSpec(BaseModel):
    name: str
    type: str = Field(..., description="e.g., Conv2d, TransformerEncoderLayer, Linear")
    params: Dict[str, Any] = Field(default_factory=dict)

class ModelSpec(BaseModel):
    architecture: str = Field(..., description="e.g., ResNet50, ViT-B/16, BERT-base")
    task_type: Optional[str] = Field(None, description="e.g., image-classification, sequence-classification")
    layers: List[LayerSpec] = Field(default_factory=list)
    loss: Optional[str] = None
    optimizer: Optional[str] = None

class HyperParams(BaseModel):
    learning_rate: Optional[float] = Field(None, gt=0)
    batch_size: Optional[int] = Field(None, gt=0)
    epochs: Optional[int] = Field(None, gt=0)
    weight_decay: Optional[float] = Field(None, ge=0)

class DatasetSpec(BaseModel):
    name: Optional[str] = None
    train_split: Optional[str] = None
    val_split: Optional[str] = None
    test_split: Optional[str] = None
    input_size: Optional[List[int]] = None  # [C,H,W] or [seq_len]
    num_classes: Optional[int] = None

class IR(BaseModel):
    paper_id: str
    title: str
    domain: Domain
    task: Optional[str] = None
    model: ModelSpec
    hyperparameters: HyperParams = Field(default_factory=HyperParams)
    dataset: DatasetSpec = Field(default_factory=DatasetSpec)
    expected_metrics: List[MetricSpec] = Field(default_factory=list)
    notes: Optional[str] = None
    uncertain: List[str] = Field(default_factory=list)

    @validator("title")
    def _title_non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("title empty")
        return v

    @validator("model")
    def _arch_non_empty(cls, v: ModelSpec):
        if not v.architecture or not v.architecture.strip():
            raise ValueError("model.architecture empty")
        return v
