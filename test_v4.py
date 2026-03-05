"""
test_v4.py  -  quick sanity check for V4 asymmetric architecture + curriculum fixes.
Run from D:\\BIO:  python test_v4.py
"""
import sys
sys.path.insert(0, ".")

import torch
import config

config.MODEL["input_dim"] = 10
from src.model import BiogasTransferModel, latent_distillation_loss

m = BiogasTransferModel()

# ---- encoder interface consistency ----
print("Encoder out_dims (all should match for shared head compatibility):")
out_m = m.encoder_muscatine.out_dim
out_d = m.encoder_dataone.out_dim
out_e = m.encoder_edi.out_dim
print(f"  Muscatine : {out_m}")
print(f"  DataONE   : {out_d}")
print(f"  EDI       : {out_e}")
assert out_m == out_d == out_e, \
    f"Output dims must match for shared head! {out_m} {out_d} {out_e}"
print("  PASS: all encoders share same out_dim (required for shared pred head)")

# ---- internal capacity reduction check ----
print("\nInternal capacity (small_out vs out_dim):")
# DataONE: small_h = hidden_dim//2, small_out = small_h//2
# Out should be larger than internal bottleneck
d_small = getattr(m.encoder_dataone, "_small_out", None)
e_small = getattr(m.encoder_edi, "_small_out", None)
print(f"  DataONE  internal bottleneck: {d_small}  VS  final out: {out_d}")
print(f"  EDI      internal bottleneck: {e_small}  VS  final out: {out_e}")
if d_small is not None:
    assert d_small < out_d, "DataONE not compressed!"
    print("  PASS: DataONE encoder is capacity-reduced (smaller internal bottleneck)")
if e_small is not None:
    assert e_small < out_e, "EDI not compressed!"
    print("  PASS: EDI encoder is capacity-reduced (smaller internal bottleneck)")

# ---- distillation loss ----
print("\nDistillation loss test:")
x = torch.randn(4, 24, 10)
z_m = m._encode(x, "muscatine")
z_d = m._encode(x, "dataone")
z_e = m._encode(x, "edi")
print(f"  z_muscatine shape: {tuple(z_m.shape)}")
print(f"  z_dataone shape:   {tuple(z_d.shape)}")
print(f"  z_edi shape:       {tuple(z_e.shape)}")
assert z_m.shape == z_d.shape == z_e.shape, "Latent shapes must match for distillation!"
dl = latent_distillation_loss(z_d, z_m)
print(f"  Distillation loss: {dl.item():.4f}")
assert not torch.isnan(dl), "Distillation loss is NaN!"
print("  PASS: distillation loss computed successfully (shapes match + finite)")

# ---- freeze / unfreeze ----
print("\nFreeze/unfreeze helpers:")
m.freeze_muscatine_encoder(verbose=False)
n_frozen = sum(1 for p in m.encoder_muscatine.parameters() if not p.requires_grad)
assert n_frozen > 0, "Freeze failed: no params are frozen"
m.unfreeze_muscatine_encoder(verbose=False)
n_unf = sum(1 for p in m.encoder_muscatine.parameters() if p.requires_grad)
assert n_unf > 0, "Unfreeze failed"
print(f"  freeze_muscatine_encoder: {n_frozen} params frozen")
print(f"  unfreeze_muscatine_encoder: {n_unf} params active again")
print("  PASS: freeze/unfreeze helpers work correctly")

# ---- predict_lodo ----
print("\npredict_lodo:")
m.eval()
with torch.no_grad():
    mean, aleat, epist = m.predict_lodo(x, excluded_domain="muscatine")
print(f"  mean shape: {tuple(mean.shape)}  (expected: {(4,)} or broadcastable)")
assert not torch.isnan(mean).any(), "predict_lodo returned NaN!"
print("  PASS: predict_lodo with Muscatine excluded (uses as prior)")

with torch.no_grad():
    mean2, _, _ = m.predict_lodo(x, excluded_domain="edi")
print(f"  mean shape (excluded=edi): {tuple(mean2.shape)}")
assert not torch.isnan(mean2).any()
print("  PASS: predict_lodo with EDI excluded")

# ---- lr_per_domain prefix match (BUG FIX CHECK) ----
print("\nlr_per_domain prefix validation (v4 bug fix):")
lr_per_domain = config.TRAIN.get("lr_per_domain", {})
for prefix, lr in lr_per_domain.items():
    matched = [n for n, _ in m.named_parameters() if n.startswith(prefix)]
    print(f"  '{prefix}' @ lr={lr}: {len(matched)} params matched")
    assert len(matched) > 0, \
        f"CRITICAL: No params matched prefix '{prefix}'! LR grouping broken."
print("  PASS: all lr_per_domain prefixes match model parameter names")

# ---- CURRICULUM config ----
print("\nCURRICULUM config:")
assert hasattr(config, "CURRICULUM"), "CURRICULUM block missing from config.py"
c = config.CURRICULUM
assert c["phase1_epochs"] == 50
assert c["phase2_epochs"] == 30
assert c["phase3_epochs"] == 40
total = c["phase1_epochs"] + c["phase2_epochs"] + c["phase3_epochs"]
print(f"  Phases: P1={c['phase1_epochs']}, P2={c['phase2_epochs']}, P3={c['phase3_epochs']}")
print(f"  Total: {total} epochs  (config source_epochs={config.TRAIN['source_epochs']})")
assert total == config.TRAIN["source_epochs"], \
    f"Phase total {total} != source_epochs {config.TRAIN['source_epochs']}"
print("  PASS: CURRICULUM phases sum to source_epochs")

# ---- SMALL_DOMAIN_HIDDEN and DISTILLATION_WEIGHT ----
print("\nV4 constants:")
assert hasattr(config, "SMALL_DOMAIN_HIDDEN"), "SMALL_DOMAIN_HIDDEN missing!"
assert config.SMALL_DOMAIN_HIDDEN == 64
assert hasattr(config, "DISTILLATION_WEIGHT"), "DISTILLATION_WEIGHT missing!"
print(f"  SMALL_DOMAIN_HIDDEN = {config.SMALL_DOMAIN_HIDDEN}")
print(f"  DISTILLATION_WEIGHT = {config.DISTILLATION_WEIGHT}")
print("  PASS: V4 constants present")

print("\n" + "="*55)
print("ALL TESTS PASSED — V4 architecture verified!")
print("="*55)
