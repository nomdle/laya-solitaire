"""
Laya Decision Agent - Option-Scoring LoRA Fine-Tuning (v3)
==========================================================
Fine-tunes `laya.common.DecisionModel` (loaded through `laya.Agent`) so that it
picks the best option out of a list.

Key points vs. the previous revision
  * option markers are located via character offsets (no guessing on '-' tokens)
  * samples whose markers/target cannot be located are SKIPPED (and counted)
  * padded markers are masked out of the softmax
  * pad_token_id is injected into the collator (not read from a dataset column)
  * LoRA target modules are discovered from the real model (no guessed names)
  * validation split is grouped by `game_id` when the dataset has it
  * Trainer receives per-batch argmax only (no ragged logits concatenation)
  * run metadata is saved so inference can rebuild the exact same prompt format
"""

import os
import sys
import json
import random
import argparse
import inspect
from collections import Counter
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import laya
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import EvalPrediction, Trainer, TrainingArguments, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput

MASK_VALUE = -1e4  # fp16-safe "minus infinity" for padded option slots


# ----------------------------------------------------------------------------
# Prompt / marker construction  (reuse this function in your inference code!)
# ----------------------------------------------------------------------------
def build_prompt(context: str, options: List[str]) -> Tuple[str, List[int]]:
    """
    Returns (prompt_text, char_positions_of_each_option_dash).
    The answer is NOT part of the prompt.
    """
    text = f"### Context:\n{context}\n\n### Options:\n"
    char_pos = []
    for i, opt in enumerate(options):
        char_pos.append(len(text))  # index of the '-' that starts this option
        text += f"- {opt}"
        if i < len(options) - 1:
            text += "\n"
    return text, char_pos


def find_target_index(options: List[str], target: Any) -> Tuple[Optional[int], str]:
    """Returns (index, status). status in {'ok', 'dup', 'missing', 'bad_index'}."""
    if isinstance(target, int) and not isinstance(target, bool):
        return (target, "ok") if 0 <= target < len(options) else (None, "bad_index")
    t = str(target).strip()
    hits = [i for i, o in enumerate(options) if str(o).strip() == t]
    if not hits:
        return None, "missing"
    return hits[0], ("dup" if len(hits) > 1 else "ok")


def build_sample(record: Dict[str, Any], tokenizer, qtype_val: int, max_length: int, shuffle_options: bool = True):
    """Returns (sample_dict, None) or (None, skip_reason)."""
    options = [str(o) for o in record.get("options", [])]
    if len(options) < 2:
        return None, "too_few_options"

    target_val = record.get("target", "")

    # 위치 편향(Position Bias)을 방지하기 위해 옵션 목록을 무작위로 섞음
    if shuffle_options:
        random.shuffle(options)

    target_idx, status = find_target_index(options, target_val)
    if target_idx is None:
        return None, f"target_{status}"

    prompt, char_pos = build_prompt(str(record.get("context", "")), options)
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]

    marker_pos: List[int] = []
    for c in char_pos:
        hit = next((t for t, (s, e) in enumerate(offsets) if e > s and s <= c < e), None)
        if hit is None:
            return None, "truncated_markers"  # an option was cut off by max_length
        marker_pos.append(hit)

    if len(set(marker_pos)) != len(marker_pos):
        return None, "marker_collision"

    group = record.get("game_id", None)
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "marker_pos": marker_pos,
        "qtype": qtype_val,
        "labels": target_idx,
        "group": None if group is None else str(group),
        "dup_target": status == "dup",
    }, None


def resolve_qtype() -> int:
    q = getattr(laya, "QTYPES", None)
    if isinstance(q, dict):
        if "choice" not in q:
            raise KeyError(f"'choice' not found in laya.QTYPES (available: {list(q)})")
        return q["choice"]
    print("[Warn] laya.QTYPES not found; using qtype=0")
    return 0


# ----------------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------------
def extract_logits(out: Any) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        return out["logits"]
    if hasattr(out, "logits"):
        return out.logits
    return out


class LayaDecisionWrapper(nn.Module):
    """
    Wraps DecisionModel for Trainer/PEFT.
    NOTE: forward() intentionally has NO **kwargs, so Trainer does not assume the
    model handles `num_items_in_batch` itself.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_pos=marker_pos,
            marker_mask=marker_mask,
            qtype=qtype,
        )
        logits = extract_logits(out)

        if logits.dim() != 2 or logits.shape != marker_mask.shape:
            raise ValueError(
                f"Expected option logits of shape {tuple(marker_mask.shape)} "
                f"[batch, num_markers], got {tuple(logits.shape)}. "
                "Check DecisionModel's output format."
            )

        logits = logits.float().masked_fill(~marker_mask, MASK_VALUE)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ----------------------------------------------------------------------------
# Collator / metrics
# ----------------------------------------------------------------------------
def collate(features: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    bsz = len(features)
    max_len = max(len(f["input_ids"]) for f in features)
    max_m = max(len(f["marker_pos"]) for f in features)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    marker_pos = torch.zeros((bsz, max_m), dtype=torch.long)
    marker_mask = torch.zeros((bsz, max_m), dtype=torch.bool)
    qtype = torch.zeros((bsz,), dtype=torch.long)
    labels = torch.zeros((bsz,), dtype=torch.long)

    for i, f in enumerate(features):
        n = len(f["input_ids"])
        m = len(f["marker_pos"])
        input_ids[i, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = torch.tensor(f["attention_mask"], dtype=torch.long)
        marker_pos[i, :m] = torch.tensor(f["marker_pos"], dtype=torch.long)
        marker_mask[i, :m] = True
        qtype[i] = f["qtype"]
        labels[i] = f["labels"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
        "labels": labels,
    }


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)  # [batch]; avoids ragged [batch, num_markers] gathering


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds = eval_pred.predictions
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    labels = eval_pred.label_ids
    return {"accuracy": float(np.mean(np.asarray(preds) == np.asarray(labels)))}


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[Warn] line {ln}: invalid JSON ({e}); skipped")
    return records


def split_samples(samples: List[Dict[str, Any]], val_ratio: float, seed: int):
    """Group-aware split when every sample has a game_id; random otherwise."""
    rng = random.Random(seed)
    n = len(samples)
    if n < 10 or val_ratio <= 0:
        return samples, []

    if all(s["group"] is not None for s in samples):
        groups = sorted({s["group"] for s in samples})
        if len(groups) >= 2:
            rng.shuffle(groups)
            n_val_groups = max(1, round(len(groups) * val_ratio))
            val_groups = set(groups[:n_val_groups])
            train = [s for s in samples if s["group"] not in val_groups]
            val = [s for s in samples if s["group"] in val_groups]
            if train and val:
                print(f"[Split] grouped by game_id: {len(groups) - n_val_groups} / {n_val_groups} games")
                return train, val

    print("[Split] no usable game_id -> random split "
          "(consecutive states of one game are similar, so val accuracy may be optimistic)")
    idx = list(range(n))
    rng.shuffle(idx)
    n_val = max(1, round(n * val_ratio))
    val_idx = set(idx[:n_val])
    train = [samples[i] for i in range(n) if i not in val_idx]
    val = [samples[i] for i in range(n) if i in val_idx]
    return train, val


def print_dataset_stats(samples, skipped: Counter, total: int):
    kept = len(samples)
    print("\n[Dataset stats]")
    print(f"  total records : {total}")
    print(f"  kept          : {kept}")
    print(f"  skipped       : {sum(skipped.values())} {dict(skipped)}")
    if not kept:
        return
    lens = np.array([len(s["input_ids"]) for s in samples])
    n_opts = np.array([len(s["marker_pos"]) for s in samples])
    labels = np.array([s["labels"] for s in samples])
    print(f"  token length  : mean={lens.mean():.0f} p95={np.percentile(lens, 95):.0f} max={lens.max()}")
    print(f"  #options      : min={n_opts.min()} mean={n_opts.mean():.1f} max={n_opts.max()}")
    print(f"  label (index) distribution: {dict(sorted(Counter(labels.tolist()).items()))}")
    print(f"  baseline acc  : random={np.mean(1.0 / n_opts):.3f} "
          f"| always-pick-most-common-index={Counter(labels.tolist()).most_common(1)[0][1] / kept:.3f}")
    dup = sum(1 for s in samples if s.get("dup_target"))
    if dup:
        print(f"  [Warn] {dup} samples have duplicated option text equal to the target (first match used)")
    if sum(skipped.values()) / max(total, 1) > 0.05:
        print("  [Warn] >5% of samples were skipped; consider raising --max_length")


# ----------------------------------------------------------------------------
# LoRA module discovery
# ----------------------------------------------------------------------------
def discover_lora_modules(model: nn.Module, head_names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (target_modules, modules_to_save) as FULL module names of `model`.
    - every nn.Linear outside the head(s) becomes a LoRA target
    - modules whose path contains a head name are fully trained (modules_to_save)
    """
    targets, heads = [], []
    for name, module in model.named_modules():
        if not name:
            continue
        parts = name.split(".")
        in_head = any(p in head_names for p in parts)
        if in_head:
            # keep only the outermost head module
            if parts[-1] in head_names:
                heads.append(name)
            continue
        if isinstance(module, nn.Linear):
            targets.append(name)
    return targets, heads


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Laya Solitaire Agent Fine-Tuning")
    p.add_argument("--model_name", type=str, default="convaiinnovations/laya")
    p.add_argument("--dataset_path", type=str, default="dataset/laya_dataset.jsonl")
    p.add_argument("--output_dir", type=str, default="./checkpoint-laya-solitaire")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--head_modules", type=str, default="scorer",
                   help="comma-separated names of the option-scoring head(s) to train fully "
                        "(check the module list printed on startup; '' to disable)")
    p.add_argument("--target_modules", type=str, default="",
                   help="comma-separated FULL module names for LoRA (default: auto-discover all nn.Linear outside heads)")
    p.add_argument("--dry_run", action="store_true", help="Validate data/modules and exit")
    args = p.parse_args()

    set_seed(args.seed)

    print("=" * 60)
    print(" Laya Decision Agent - Option Scoring LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Model:   {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output:  {args.output_dir}")

    if not os.path.exists(args.dataset_path):
        print(f"\n[Error] Dataset not found: {args.dataset_path}")
        print("Run 'python game_engine.py --headless --collect 100' first.")
        sys.exit(1)

    records = load_jsonl(args.dataset_path)
    if not records:
        print("[Error] Dataset is empty.")
        sys.exit(1)

    # 1. Model / tokenizer ----------------------------------------------------
    print(f"\nLoading Laya Agent: '{args.model_name}' ...")
    agent = laya.Agent(args.model_name)
    base_model = agent.model
    tokenizer = agent.tok

    if not getattr(tokenizer, "is_fast", False):
        print("[Error] A fast tokenizer is required (offset mapping).")
        sys.exit(1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        print("[Error] Tokenizer has neither pad nor eos token id.")
        sys.exit(1)

    qtype_val = resolve_qtype()

    # 2. Preprocess -----------------------------------------------------------
    print("Building samples and locating option markers ...")
    samples, skipped = [], Counter()
    for rec in records:
        sample, reason = build_sample(rec, tokenizer, qtype_val, args.max_length)
        if sample is None:
            skipped[reason] += 1
        else:
            samples.append(sample)
    print_dataset_stats(samples, skipped, len(records))

    if not samples:
        print("[Error] No usable samples after preprocessing.")
        sys.exit(1)

    # 3. Wrap + discover LoRA modules ----------------------------------------
    model_wrapped = LayaDecisionWrapper(base_model)
    head_names = [h.strip() for h in args.head_modules.split(",") if h.strip()]
    if args.target_modules.strip():
        target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        _, heads = discover_lora_modules(model_wrapped, head_names)
    else:
        target_modules, heads = discover_lora_modules(model_wrapped, head_names)

    print(f"\n[LoRA] {len(target_modules)} target Linear modules "
          f"(e.g. {target_modules[:3]}{' ...' if len(target_modules) > 3 else ''})")
    print(f"[LoRA] fully-trained head modules: {heads if heads else 'NONE'}")
    if head_names and not heads:
        print(f"[Warn] head name(s) {head_names} not found in the model -> the scoring head stays FROZEN.\n"
              "       Top-level modules are:")
        for n, _ in base_model.named_children():
            print(f"         - {n}")
    if not target_modules:
        print("[Error] No nn.Linear modules found for LoRA. Pass --target_modules explicitly.")
        sys.exit(1)

    if args.dry_run:
        s = samples[0]
        print("\n[Dry Run] first sample:")
        print(f"  tokens={len(s['input_ids'])} markers={s['marker_pos']} label={s['labels']}")
        print("[Dry Run] done. Exiting without training.")
        sys.exit(0)

    # 4. Split + datasets -----------------------------------------------------
    train_s, val_s = split_samples(samples, args.val_ratio, args.seed)
    keep = ["input_ids", "attention_mask", "marker_pos", "qtype", "labels"]
    to_ds = lambda xs: Dataset.from_list([{k: x[k] for k in keep} for x in xs])
    train_ds = to_ds(train_s)
    eval_ds = to_ds(val_s) if val_s else None
    print(f"Train={len(train_ds)}  Val={len(eval_ds) if eval_ds is not None else 0}")

    # 5. LoRA -----------------------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=heads or None,
        bias="none",
    )
    model = get_peft_model(model_wrapped, peft_config)
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        print("[Error] 0 trainable parameters.")
        sys.exit(1)

    # 6. Training -------------------------------------------------------------
    use_cuda = torch.cuda.is_available()
    param_dtype = next(base_model.parameters()).dtype
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and (not use_bf16) and param_dtype == torch.float32
    print(f"Precision: bf16={use_bf16} fp16={use_fp16} (param dtype={param_dtype})")

    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 4),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        label_names=["labels"],
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    if eval_ds is not None:
        strat_key = ("eval_strategy"
                     if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
                     else "evaluation_strategy")
        ta_kwargs[strat_key] = "epoch"
        ta_kwargs.update(load_best_model_at_end=True, metric_for_best_model="accuracy", greater_is_better=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=partial(collate, pad_token_id=pad_token_id),
        compute_metrics=compute_metrics if eval_ds is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_ds is not None else None,
    )
    # The wrapper computes a plain per-batch mean loss; don't let Trainer rescale it.
    trainer.model_accepts_loss_kwargs = False

    print("\nStarting LoRA fine-tuning ...")
    trainer.train()

    # 7. Save -----------------------------------------------------------------
    print(f"\nSaving adapter to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    meta = {
        "base_model": args.model_name,
        "max_length": args.max_length,
        "qtype": qtype_val,
        "prompt_format": "### Context:\\n{context}\\n\\n### Options:\\n- {opt}\\n- {opt}...",
        "marker": "token covering the '-' that starts each option (via offset mapping)",
        "head_modules": heads,
        "target_modules": target_modules,
        "mask_value": MASK_VALUE,
    }
    with open(os.path.join(args.output_dir, "laya_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
    