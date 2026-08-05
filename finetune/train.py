"""Fine-tune Qwen2.5-14B on MedMCQA. Method (full | lora | qlora) is the only knob.

Launch: torchrun --nproc_per_node=N train.py --config configs/<method>.yaml [--out DIR]
(single-GPU methods run under plain `python`; see modal_app.py).
"""
import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    set_seed,
)

import data
from bench import Benchmark, BenchmarkCallback, BenchTrainer


def load_model(cfg):
    if cfg["method"] == "qlora":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], quantization_config=quant, torch_dtype=torch.bfloat16
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # reentrant GC breaks under any DDP wrap
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.bfloat16)

    if cfg["method"] in ("lora", "qlora"):
        lora = cfg["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora["r"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                target_modules=lora["target_modules"],
                task_type="CAUSAL_LM",
            ),
        )
        model.enable_input_require_grads()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    assert torch.cuda.is_available(), "no CUDA device"
    set_seed(cfg["seed"])
    out_dir = str(Path(args.out) / cfg["name"])

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = data.train_dataset(
        tok, n=cfg["train_size"], max_len=cfg["max_len"], seed=cfg["seed"]
    )
    model = load_model(cfg)

    bench = Benchmark()
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["per_device_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["lr"]),
        num_train_epochs=cfg["epochs"],
        bf16=True,
        gradient_checkpointing=cfg["method"] != "qlora",  # qlora enables it via prepare_model_for_kbit_training
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="no",
        deepspeed=cfg["deepspeed"],
        report_to="none",
        run_name=cfg["name"],
    )
    meta = {"method": cfg["name"], "model_id": cfg["model_id"]}

    trainer = BenchTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        callbacks=[BenchmarkCallback(bench, out_dir, meta, warmup=8)],
        bench=bench,
    )
    trainer.train()

    trainer.save_model(out_dir)  
    if trainer.is_world_process_zero():
        tok.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
