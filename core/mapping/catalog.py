# core/mapping/catalog.py
from typing import Dict, Any, Optional

def _base_mapping() -> Dict[str, Any]:
    return {
        "imports": [],
        "framework": None,
        "model_constructor": None,
        "loss_constructor": None,
        "optimizer_constructor": None,
        "preprocessing": {"image": None, "text_tokenizer": None},
        "notes": [],
    }

def for_vit(ir: Dict[str, Any]) -> Dict[str, Any]:
    out = _base_mapping()
    out["framework"] = "transformers"
    out["imports"] = [
        "import torch",
        "from transformers import ViTForImageClassification, ViTImageProcessor",
    ]
    num_classes = ir.get("dataset", {}).get("num_classes") or 1000
    out["model_constructor"] = (
        f'ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels={num_classes})'
    )
    out["preprocessing"]["image"] = 'ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")'
    out["loss_constructor"] = "torch.nn.CrossEntropyLoss()"
    opt = (ir.get("model", {}) or {}).get("optimizer") or "AdamW"
    lr = ir.get("hyperparameters", {}).get("learning_rate") or 5e-5
    wd = ir.get("hyperparameters", {}).get("weight_decay") or 0.0
    out["optimizer_constructor"] = f'torch.optim.{ "AdamW" if str(opt).lower()=="adamw" else "Adam"}(model.parameters(), lr={lr}, weight_decay={wd})'
    out["notes"].append("Mapped ViT to HF ViTForImageClassification (catalog).")
    return out

def for_bert_sequence_classification(ir: Dict[str, Any]) -> Dict[str, Any]:
    out = _base_mapping()
    out["framework"] = "transformers"
    out["imports"] = [
        "import torch",
        "from transformers import BertForSequenceClassification, AutoTokenizer",
    ]
    num_classes = ir.get("dataset", {}).get("num_classes") or 2
    out["model_constructor"] = f'BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels={num_classes})'
    out["preprocessing"]["text_tokenizer"] = 'AutoTokenizer.from_pretrained("bert-base-uncased")'
    out["loss_constructor"] = "torch.nn.CrossEntropyLoss()"
    lr = ir.get("hyperparameters", {}).get("learning_rate") or 2e-5
    wd = ir.get("hyperparameters", {}).get("weight_decay") or 0.0
    out["optimizer_constructor"] = f'torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay={wd})'
    out["notes"].append("Mapped BERT to HF BertForSequenceClassification (catalog).")
    return out

def for_resnet50(ir: Dict[str, Any]) -> Dict[str, Any]:
    out = _base_mapping()
    out["framework"] = "pytorch"
    out["imports"] = [
        "import torch",
        "import torchvision",
        "import torchvision.transforms as T",
    ]
    num_classes = ir.get("dataset", {}).get("num_classes") or 1000
    # pick a pretrained weight name compatible with new torchvision versions; user can change later
    out["model_constructor"] = f'torchvision.models.resnet50(weights="IMAGENET1K_V2")'
    out["notes"].append("Replace final FC for custom num_classes if needed.")
    out["loss_constructor"] = "torch.nn.CrossEntropyLoss()"
    lr = ir.get("hyperparameters", {}).get("learning_rate") or 1e-3
    wd = ir.get("hyperparameters", {}).get("weight_decay") or 1e-4
    out["optimizer_constructor"] = f'torch.optim.Adam(model.parameters(), lr={lr}, weight_decay={wd})'
    out["preprocessing"]["image"] = "T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])"
    return out

def for_xgboost_classifier(ir: Dict[str, Any]) -> Dict[str, Any]:
    out = _base_mapping()
    out["framework"] = "xgboost"
    out["imports"] = [
        "import xgboost as xgb",
        "from sklearn.metrics import accuracy_score",
        "from sklearn.model_selection import train_test_split",
    ]
    lr = ir.get("hyperparameters", {}).get("learning_rate") or 0.1
    out["model_constructor"] = f'xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate={lr}, subsample=0.8, colsample_bytree=0.8, tree_method="hist")'
    out["notes"].append("Using XGBClassifier with reasonable defaults (catalog).")
    return out

def choose_by_ir(ir: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    arch = (ir.get("model", {}) or {}).get("architecture") or ""
    domain = (ir.get("domain") or "").lower()
    task_type = (ir.get("model", {}) or {}).get("task_type") or ""

    arch_l = str(arch).lower()
    task_l = str(task_type).lower()

    if "vit" in arch_l or "vision transformer" in arch_l or (domain == "cv" and "transformer" in task_l):
        return for_vit(ir)
    if "bert" in arch_l:
        return for_bert_sequence_classification(ir)
    if "resnet" in arch_l:
        return for_resnet50(ir)
    if "xgboost" in arch_l or (domain == "ml" and "boost" in task_l):
        return for_xgboost_classifier(ir)
    return None
