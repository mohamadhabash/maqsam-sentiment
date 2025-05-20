#!/usr/bin/env python3
"""
Benchmark Inference Time & GPU Memory using ModelLoader and SentimentPredictor
"""

import time
import torch
from app.model_loader import ModelLoader
from app.predictor import SentimentPredictor

def human_gb(bytes_amt: int) -> float:
    return bytes_amt / (1024 ** 3)

def main():
    # 1) Initialize loader and predictor
    loader = ModelLoader("config/config.yaml")
    device = loader.get_device()
    print(f">>> Using device: {device}")

    predictor = SentimentPredictor()  # loads tokenizer & model internally
    model = predictor.model  # already on device
    tokenizer = predictor.tokenizer
    cfg = loader.cfg  # full config dict

    # 2) Measure baseline GPU memory (weights only)
    if device.type == "cuda":
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated()
        print(f"Baseline GPU memory (weights only): {human_gb(base_mem):.3f} GB")
    else:
        print("Note: running on CPU—GPU memory not measured.")

    # 3) Build a sample prompt
    prompt_base = cfg["prompt"]["base"]
    task_rules  = cfg["task"]["sentiment"]
    sample_text = (
        "The customer was experiencing an issue with the laptop that has not been resolved yet "
        "after several attempts, and an appointment was scheduled to follow up on the case next Sunday"
    )
    question    = f"{task_rules}\n\nText: \"{sample_text}\""
    full_prompt = prompt_base.format_map({"Question": question})

    # 4) Prepare inputs
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # 5) Warm-up (first generate includes JIT/cache overhead)
    print("Warming up …")
    _ = model.generate(
        **inputs,
        max_new_tokens=cfg["model"]["max_length"],
        temperature=cfg["model"]["temperature"],
        top_p=cfg["model"]["top_p"],
        repetition_penalty=cfg["model"]["repetition_penalty"],
        do_sample=False
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 6) Benchmark latency & peak memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    _ = model.generate(
        **inputs,
        max_new_tokens=cfg["model"]["max_length"],
        temperature=cfg["model"]["temperature"],
        top_p=cfg["model"]["top_p"],
        repetition_penalty=cfg["model"]["repetition_penalty"],
        do_sample=False
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = (time.time() - start) * 1000.0  # ms

    print(f"Inference latency (1 request): {latency:.1f} ms")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory during generation: {human_gb(peak):.3f} GB")


if __name__ == "__main__":
    main()
