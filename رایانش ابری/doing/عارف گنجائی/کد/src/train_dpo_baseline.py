import os
import random
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


@dataclass
class RunCfg:
    model_name: str = "gpt2"
    dataset_name: str = "Anthropic/hh-rlhf"
    subset_train: int = 200   # سبک
    subset_eval: int = 50     # سبک
    beta: float = 0.1         # β ثابت (Baseline)
    max_prompt_len: int = 256
    max_length: int = 512     # قبلاً 384 بود؛ کمی بیشتر ولی هنوز امن و سبک
    out_dir: str = "runs/dpo_baseline_gpt2"


def _pick(split, n, seed=42):
    idx = list(range(len(split)))
    random.Random(seed).shuffle(idx)
    return split.select(idx[:n])


def main():
    cfg = RunCfg()
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 1) dataset: HH
    ds = load_dataset(cfg.dataset_name)
    train = _pick(ds["train"], cfg.subset_train, seed=1)
    evals = _pick(ds["test"], cfg.subset_eval, seed=2)

    # نرمال‌سازی: خروجی حتماً prompt/chosen/rejected داشته باشد
    def normalize(ex):
        prompt = ex.get("prompt", "")
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    train = train.map(normalize, remove_columns=train.column_names)
    evals = evals.map(normalize, remove_columns=evals.column_names)

    # 2) model + tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    # 3) TRL config (CPU-friendly)
    dpo_args = DPOConfig(
        output_dir=cfg.out_dir,
        beta=cfg.beta,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=1,
        logging_steps=1,
        eval_steps=25,
        save_steps=50,
        max_prompt_length=cfg.max_prompt_len,
        max_length=cfg.max_length,

        # مهم برای GPT2 (جلوگیری از خطای طول زیاد)
        truncation_mode="keep_end",

        fp16=False,
        bf16=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train,
        eval_dataset=evals,
        processing_class=tok,
    )

    # ---- FIX 1: اطمینان از اینکه input_ids/labels حتماً Long باشند (نه Float) ----
    orig_collator = trainer.data_collator

    def safe_collator(features):
        batch = orig_collator(features)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and ("input_ids" in k or "labels" in k):
                batch[k] = v.long()
        return batch

    trainer.data_collator = safe_collator
    # ---------------------------------------------------------------------------

    trainer.train()
    trainer.save_model(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)

    print(f"\n✅ Done. Logs/Model saved in: {cfg.out_dir}")


if __name__ == "__main__":
    main()