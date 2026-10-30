# Colab Stage 1 - copy-paste cells

Train the stance model's general skill (Stage 1) on SemEval-2016 Task 6A. Paste each block into its own Colab cell, run it, check the note under it. Cells 1-5 = setup + data; cells 6-10 = train + evaluate + save. These cells were smoke-tested on CPU with a tiny model against transformers 5.19.0 / datasets 5.1.0; the real run on a T4 is the first time they touch real data.

## Cell 1 - Check the GPU

```python
!nvidia-smi -L
import torch
print("CUDA available:", torch.cuda.is_available())
```

> Should print a Tesla T4 and `CUDA available: True`. If not: Runtime -> Change runtime type -> T4 GPU, then re-run.

## Cell 2 - Install / check libraries

```python
!pip install -q transformers datasets accelerate scikit-learn
import transformers, datasets, sklearn, torch
print("transformers", transformers.__version__, "| datasets", datasets.__version__, "| torch", torch.__version__)
```

> Tested on transformers 5.19.0 + datasets 5.1.0 + torch 2.14. If pip asks you to restart the runtime, restart, then re-run from cell 1 (cell 2 will be quick).

## Cell 3 - Mount Google Drive (where results are saved)

```python
from google.colab import drive
drive.mount("/content/drive")

import os
OUT_DIR = "/content/drive/MyDrive/econsult"
os.makedirs(OUT_DIR, exist_ok=True)
print("results will be saved to:", OUT_DIR)
```

> Click through the Google permission popup. Drive is the only thing that survives when Colab disconnects.

## Cell 4 - Load Stage 1 data (SemEval-2016 Task 6A via TweetEval)

```python
import pandas as pd
from datasets import load_dataset

# Stage 1 data: the five stance subsets of SemEval-2016 Task 6A, as packaged in TweetEval.
TARGETS = {
    "stance_abortion": "Legalization of Abortion",
    "stance_atheism": "Atheism",
    "stance_climate": "Climate Change is a Real Concern",
    "stance_feminist": "Feminist Movement",
    "stance_hillary": "Hillary Clinton",
}
# Label ids exactly as TweetEval defines them: 0 = none, 1 = against, 2 = favor
ID2LABEL = {0: "NONE", 1: "AGAINST", 2: "FAVOR"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

frames = {"train": [], "validation": [], "test": []}
for cfg, target in TARGETS.items():
    ds = load_dataset("cardiffnlp/tweet_eval", cfg)
    for split in frames:
        d = ds[split].to_pandas()
        d["target"] = target      # what the tweet takes a stance ON
        d["config"] = cfg
        frames[split].append(d)

train_df = pd.concat(frames["train"], ignore_index=True)
val_df = pd.concat(frames["validation"], ignore_index=True)
test_df = pd.concat(frames["test"], ignore_index=True)
print("loaded:", len(train_df), "train,", len(val_df), "val,", len(test_df), "test")
```

> Takes under a minute. A Hugging Face token warning is harmless.

## Cell 5 - Look at the data before trusting it

```python
print("train / val / test:", len(train_df), "/", len(val_df), "/", len(test_df))
print("expected:           2620 /  294 / 1249\n")

for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(name, df["label"].map(ID2LABEL).value_counts().to_dict())

print("\nTweets per target (train):")
print(train_df["target"].value_counts().to_string())

print("\n5 random training examples:")
for _, r in train_df.sample(5, random_state=1).iterrows():
    print(f"[{ID2LABEL[r['label']]:7}] target = {r['target']}")
    print("   ", r["text"], "\n")
```

> Compare the printed sizes with the expected line.

## Cell 6 - Tokenise (turn text into numbers)

```python
import torch
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAXLEN = 128
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

class StanceDataset(torch.utils.data.Dataset):
    """Holds tokenised (target, text) pairs plus labels; the Trainer pads batches on the fly."""
    def __init__(self, df):
        # Encode as a PAIR: (target, text). BERT turns this into [CLS] target [SEP] text [SEP],
        # so the model always knows WHAT the sentence is taking a stance on.
        self.enc = tok(df["target"].tolist(), df["text"].tolist(), truncation=True, max_length=MAXLEN)
        self.labels = df["label"].tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}   # input_ids, token_type_ids, attention_mask
        item["labels"] = self.labels[i]
        return item

train_ds, val_ds, test_ds = StanceDataset(train_df), StanceDataset(val_df), StanceDataset(test_df)
print("examples:", len(train_ds), "train,", len(val_ds), "val,", len(test_ds), "test")
print("\nWhat the model actually sees for example 0:")
print(tok.decode(train_ds[0]["input_ids"]))
print("true label:", ID2LABEL[train_ds[0]["labels"]])
```

## Cell 7 - Build the model and TRAIN

```python
import inspect
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorWithPadding, set_seed,
)

set_seed(42)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
)
collator = DataCollatorWithPadding(tokenizer=tok)

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(logits, axis=-1)
    y = p.label_ids
    f1s = f1_score(y, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "f1_none": float(f1s[0]),
        "f1_against": float(f1s[1]),
        "f1_favor": float(f1s[2]),
        # SemEval-2016's official score: mean of the FAVOR and AGAINST F1 only
        "f_avg_semeval": float((f1s[1] + f1s[2]) / 2),
    }

# The argument was renamed between transformers versions; pick whichever exists.
params = inspect.signature(TrainingArguments.__init__).parameters
strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"

args = TrainingArguments(
    output_dir="/content/stage1_ckpt",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42,
    logging_steps=50,
    **{strategy_key: "epoch"},
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
)
trainer.train()
```

> About 3-6 minutes on a T4. Watch the per-epoch table.

## Cell 8 - Evaluate on the held-out test set

```python
from sklearn.metrics import classification_report, confusion_matrix

val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")
test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

pred = trainer.predict(test_ds)
yt = pred.label_ids
yp = np.argmax(pred.predictions, axis=-1)

print("=== TEST set (held out, never trained on) ===")
print(classification_report(yt, yp, target_names=["NONE", "AGAINST", "FAVOR"], digits=3, zero_division=0))
print("Confusion matrix (rows = true, cols = predicted; order NONE, AGAINST, FAVOR):")
print(confusion_matrix(yt, yp, labels=[0, 1, 2]))

# Context: what would a model that ALWAYS guesses the most common training label score?
maj = int(train_df["label"].value_counts().idxmax())
base = np.full(len(yt), maj)
print(f"\nMajority-class baseline (always '{ID2LABEL[maj]}'): "
      f"accuracy {accuracy_score(yt, base):.3f}, macro-F1 {f1_score(yt, base, average='macro', zero_division=0):.3f}")
print(f"Our model:                                  "
      f"accuracy {accuracy_score(yt, yp):.3f}, macro-F1 {f1_score(yt, yp, average='macro', zero_division=0):.3f}")
print(f"SemEval-style F_avg (FAVOR/AGAINST mean):   {test_metrics['test_f_avg_semeval']:.3f}")

print("\nPer-target test macro-F1:")
tmp = test_df.copy()
tmp["pred"] = yp
for t, g in tmp.groupby("target"):
    f = f1_score(g["label"], g["pred"], average="macro", labels=[0, 1, 2], zero_division=0)
    print(f"  {t:36} n={len(g):4}  macro-F1={f:.3f}")
```

## Cell 9 - Save model + metrics to Drive

```python
import json
import transformers, datasets, sklearn

SAVE_DIR = f"{OUT_DIR}/stance_stage1_final"
trainer.save_model(SAVE_DIR)
tok.save_pretrained(SAVE_DIR)

results = {
    "stage": 1,
    "base_model": MODEL_NAME,
    "data": "cardiffnlp/tweet_eval stance_* subsets (SemEval-2016 Task 6A)",
    "labels": ID2LABEL,
    "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
    "seed": 42, "max_len": MAXLEN,
    "epochs": args.num_train_epochs, "learning_rate": args.learning_rate,
    "val": val_metrics,
    "test": test_metrics,
    "versions": {
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
    },
}
metrics_path = f"{OUT_DIR}/stance_stage1_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2, default=float)

print("saved model     ->", SAVE_DIR)
print("saved metrics   ->", metrics_path)
print(json.dumps({"test": test_metrics}, indent=2))
```

## Cell 10 - Preview on 3 legislative-style examples (demo only)

```python
from transformers import AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
reloaded = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device).eval()

def predict_stance(clause, comment):
    enc = tok(clause, comment, return_tensors="pt", truncation=True, max_length=MAXLEN).to(device)
    with torch.no_grad():
        probs = torch.softmax(reloaded(**enc).logits, dim=-1)[0].cpu().numpy()
    return {ID2LABEL[i]: round(float(p), 3) for i, p in enumerate(probs)}

# DEMONSTRATION ONLY - three hand-written examples, not data. The model has seen
# tweets about abortion/atheism/etc., never legislation, so expect some mistakes:
# that gap is exactly what Stage 2 (domain adaptation) is for.
clause = "Companies with turnover below Rs 5 crore are exempt from the new quarterly filing requirement."
examples = [
    "This exemption is essential for small businesses that cannot afford extra compliance staff.",
    "Exempting small firms will let bad actors hide behind a low turnover figure. Remove the exemption.",
    "The committee should publish the list of consulted stakeholders.",
]
for text in examples:
    print(predict_stance(clause, text), "<-", text)
```

> Expect some mistakes: the model has never seen legislation. That is what Stage 2 fixes.
