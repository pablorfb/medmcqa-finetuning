"""Evaluate a model on MedMCQA (dev split): accuracy + weighted F1. Single GPU.

  python eval_medmcqa.py --model <checkpoint-dir | hf-id> --out runs/<name> [--n 0] [--base]

Heavy deps are imported lazily so `parse_letter` stays unit-testable offline.
"""
import argparse
import json
import re
from pathlib import Path

_ANSWER = re.compile(r"ANSWER[:\s]*([ABCD])")
_LETTER = re.compile(r"[ABCD]")


def parse_letter(text: str):
    """The letter after the final 'Answer:'; fall back to the last A/B/C/D (CoT text is full of stray letters)."""
    up = text.upper()
    if m := _ANSWER.findall(up):
        return m[-1]
    m = _LETTER.findall(up)
    return m[-1] if m else None


def _load(model_path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if Path(model_path, "adapter_config.json").exists():  # LoRA/QLoRA adapter dir
        base = json.loads(Path(model_path, "adapter_config.json").read_text())["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def evaluate(model_path, out_dir, n=0, base=False, batch_size=16):
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    import data

    model, tok = _load(model_path)
    model.eval()
    rows = list(data.eval_dataset(n or None))

    preds, golds = [], []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        enc = tok([data.format_prompt(r) for r in batch], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        preds += [parse_letter(g) for g in gen]
        golds += [data.gold_letter(r) for r in batch]

    filled = [p or "?" for p in preds]  # unparseable -> guaranteed wrong
    record = {
        "name": Path(out_dir).name,
        "base": base,
        "n": len(golds),
        "none": sum(p is None for p in preds),
        "accuracy": round(accuracy_score(golds, filled), 4),
        "weighted_f1": round(f1_score(golds, filled, average="weighted", zero_division=0), 4),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "eval.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="eval subset size; 0 = full dev split")
    ap.add_argument("--base", action="store_true", help="label as the zero-shot baseline")
    args = ap.parse_args()
    evaluate(args.model, args.out, args.n, args.base)


if __name__ == "__main__":
    main()
