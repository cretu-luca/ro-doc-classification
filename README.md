# ro-doc-classification

We are given the task of finetuning a Small Language Model (SLM) for text classification, for which we use consumer hardware -- MacBook Pro 14", Apple M4 Pro (12 CPU, 16 GPU), 24 GB RAM. 

Our results show the family of models we picked, together with finetuning strategies we used represent a potential solution in real work scenarios, which feature both large and small source datasets of documents and can further be integrated within a document classification pipeline. However, we awknowledge a product-grade solution requires a larger accuracy (e.g. above 90%) and we argue such results can be accomplished by using SLM's with more parameters (e.g.`readerbench/RoBERT-large`) and better hyperparameters, together with more computing power and time resources. 

We pick the [ReaderBench RoBERT](https://huggingface.co/readerbench/RoBERT-small) family of models as backbones for our classification models and use the [MOROCO](https://aclanthology.org/P19-1068/) dataset for six news categories (culture, finance, politics, science, sports, tech), suitable for Romanian and Moldavian language tasks. 

Outputs and saved configs: `outputs/` (`outputs/<run>/config.toml` per run).

## 1. Models and adaptation

**Full fine-tuning** updates every weight in the pretrained model. **LoRA** keeps the backbone frozen and trains small low-rank matrices injected into the self-attention `query` and `value` projections (via PEFT).

Backbones: `readerbench/RoBERT-small`, `readerbench/RoBERT-base`. For full finetuning runs, we consider 3 epochs with learning rate `2e-5`. For LoRA runs, we consider 5 epochs, with hyperparamters `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, `1e-4`. Batch size 16 for all (see `config.toml`).

Based on computational cost, we consider `full-small` to be most suitable for our budget-constrained scenario. Therefore, we consider a less favorable task, with less training data to further test the limits of our model and finetuning strategy and use **20%** subset of the training split for **10 epochs** at **1e-5** (see `src/notebooks/finetune_full.ipynb`; saved under `outputs/full-data-scarce`).


| Run directory      | Backbone                   | Finetune Strategy            | Trainable params | Total params | Trainable % |
| ------------------ | -------------------------- | ---------------------------- | ---------------- | ------------ | ----------- |
| `full-small`       | `readerbench/RoBERT-small` | Full fine-tune               | 19 350 278       | 19 350 278   | 100%        |
| `full-data-scarce` | `readerbench/RoBERT-small` | Full fine-tune (data-scarce) | 19 350 278       | 19 350 278   | 100%        |
| `full-base`        | `readerbench/RoBERT-base`  | Full fine-tune               | 115 067 142      | 115 067 142  | 100%        |
| `lora-small`       | `readerbench/RoBERT-small` | LoRA                         | 99 846           | 19 450 124   | 0.51%       |
| `lora-base`        | `readerbench/RoBERT-base`  | LoRA                         | 299 526          | 115 366 668  | 0.26%       |


## 2. Test-set accuracy and inference latency

Test: 5 924 documents. Labels: **CUL**, **FIN**, **POL**, **SCI**, **SPO**, **TEC**.


| Run                            | Accuracy   | Macro F1   | Weighted F1 | Latency (ms / doc) |
| ------------------------------ | ---------- | ---------- | ----------- | ------------------ |
| `full-small`                   | 0.8589     | 0.8642     | 0.8589      | ~5.76              |
| `full-data-scarce` (20% train) | 0.8214     | 0.8182     | 0.8244      | ~5.67              |
| `full-base`                    | **0.8697** | **0.8833** | **0.8698**  | ~22.25             |
| `lora-small`                   | 0.8217     | 0.8147     | 0.8210      | ~5.72              |
| `lora-base`                    | 0.8597     | 0.8638     | 0.8600      | ~28.03             |


Latency: mean over repeated forwards on one tokenized example (`src/eval/metrics.py`, `measure_latency`). Comparable only across runs on the same hardware.

## 3. Training time and hardware


| Run                            | Train time (s) | Train time (approx.) | Train throughput (samples/s) |
| ------------------------------ | -------------- | -------------------- | ---------------------------- |
| `full-small`                   | 1 923.7        | ~32 min              | ~33.9                        |
| `full-data-scarce` (20% train) | 1 699.3        | ~28 min              | ~25.6                        |
| `full-base`                    | 6 758.7        | ~1 h 53 min          | ~9.6                         |
| `lora-small`                   | 2 987.5        | ~50 min              | ~36.3                        |
| `lora-base`                    | 9 421.7        | ~2 h 37 min          | ~11.5                        |


LoRA uses five epochs; full fine-tuning uses three, so wall time does not scale with “fewer trainable parameters” alone. Latency numbers are in section 2.

## 4. Environment

```bash
# uv
uv sync
```

Open `src/notebooks/*.ipynb` in the IDE and set the notebook **kernel / Python interpreter** to the project env (`.venv` created by `uv`, or the interpreter `uv` uses after `uv sync`).

```bash
# venv + pip (no uv)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First notebook cell `!uv pip install -r requirements.txt` is optional after `uv sync` or `pip install -r requirements.txt`.

---

