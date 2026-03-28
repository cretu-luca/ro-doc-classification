import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType


def load_model(config):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=config["model"]["num_labels"],
    )

    lc = config["training"]["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        target_modules=["query", "value"],
    )

    return get_peft_model(base_model, lora_config)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def train(model, train_dataset, val_dataset, config):
    tc = config["training"]["lora"]

    args = TrainingArguments(
        output_dir="outputs/lora",
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        learning_rate=tc["learning_rate"],
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    return trainer
