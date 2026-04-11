import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


# -------------------------------------------------
# Paths
# -------------------------------------------------
RUN_DIR = Path(r"C:\Users\theKI\Desktop\LukeActsVec_Run1_2026-04-10")
VEC_PATH = RUN_DIR / "artifacts" / "vectors" / "lukeacts_vectors_best.pt"
META_PATH = RUN_DIR / "artifacts" / "vectors" / "meta.json"
EXPORT_DIR = RUN_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Load data
# -------------------------------------------------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

names = meta["gospels"]
vectors = torch.load(VEC_PATH, weights_only=True)

# Normalize rows
vectors = vectors / vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)

# Cosine matrix
cos = vectors @ vectors.T
cos_np = cos.cpu().numpy()

print("Books:", names)
print("Vector shape:", tuple(vectors.shape))
print("Cosine matrix:\n", cos_np)


# -------------------------------------------------
# Plot heatmap
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cos_np, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_yticklabels(names)

ax.set_title("LukeActsVec Run 1 — Best-Layer Cosine Matrix")

# Add value labels
for i in range(len(names)):
    for j in range(len(names)):
        val = cos_np[i, j]
        ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color="black")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Cosine similarity")

plt.tight_layout()

# -------------------------------------------------
# Save
# -------------------------------------------------
png_path = EXPORT_DIR / "LukeActsVec_Run1_Cosine_Heatmap.png"
plt.savefig(png_path, dpi=300)
print(f"Saved heatmap to: {png_path}")

plt.show()