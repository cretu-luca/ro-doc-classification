import json
import time
import shutil
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

CATEGORY_NAMES = ["CUL", "FIN", "POL", "SCI", "SPO", "TEC"]


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} | Total: {total:,} | Percentage: {100 * trainable / total:.2f}%")
    return trainable, total


def evaluate(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=CATEGORY_NAMES)
    report_dict = classification_report(all_labels, all_preds, target_names=CATEGORY_NAMES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    print(report)
    print("Confusion matrix:")
    print(cm)

    return all_labels, all_preds, report_dict, cm


def measure_latency(model, tokenizer, sample_text, device, max_length, iterations=50):
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(**inputs)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iterations
    print(f"Average Inference Latency: {avg_ms:.2f} ms per document ({iterations} iterations)")
    return avg_ms


def save_results(output_dir, config, trainer, report_dict, cm, latency_ms, trainable_params, total_params):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_name = out.name
    model_dir = Path("models") / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy("config.toml", out / "config.toml")

    with open(out / "training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)

    train_metrics = {k: v for k, v in trainer.state.log_history[-1].items() if "train_" in k or "runtime" in k or "samples_per_second" in k or "steps_per_second" in k}
    with open(out / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    eval_results = {
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "latency_ms": latency_ms,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(100 * trainable_params / total_params, 2),
    }
    with open(out / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    trainer.save_model(str(model_dir))

    print(f"Results saved to {out}/")
    print(f"Model saved to {model_dir}/")
