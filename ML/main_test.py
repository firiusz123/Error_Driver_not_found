#!/usr/bin/env python3

import torch
import onnx
import onnxsim.simplify
from transformers import AutoModelForImageClassification
import os
os.environ['HF_TOKEN'] = ''

# ── 1. Load model only ───────────────────────────────────────────────────────
MODEL_ID = "chbh7051/driver-drowsiness-detection"

model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.eval()

# ── 2. Static dummy input ────────────────────────────────────────────────────
dummy = torch.randn(1, 3, 224, 224)

# ── 3. Export with opset=9 for MATLAB compatibility ──────────────────────────
RAW_PATH = "drowsiness_raw.onnx"

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        RAW_PATH,
        opset_version=9,           # ← corrected: safest for MATLAB
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,         # static shapes required by MATLAB
        do_constant_folding=True,
        export_params=True,
    )
print(f"Raw ONNX saved → {RAW_PATH}")

# ── 4. Simplify ──────────────────────────────────────────────────────────────
raw = onnx.load(RAW_PATH)
clean, ok = simplify(raw)
assert ok, "Simplification failed — use raw model instead"

FINAL_PATH = "drowsiness_matlab.onnx"
onnx.save(clean, FINAL_PATH)
print(f"Simplified ONNX saved → {FINAL_PATH}")

# ── 5. Validate ───────────────────────────────────────────────────────────────
onnx.checker.check_model(onnx.load(FINAL_PATH))
print("✓ Model is valid")

# ── 6. Print I/O shapes ───────────────────────────────────────────────────────
m = onnx.load(FINAL_PATH)
print("\n--- Inputs ---")
for inp in m.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape}")
print("--- Outputs ---")
for out in m.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape}")