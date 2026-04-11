import json
from pathlib import Path

import matplotlib.pyplot as plt


# -------------------------------------------------
# Paths
# -------------------------------------------------
RUN_DIR = Path(r"C:\Users\theKI\Desktop\LukeActsVec_Run1_2026-04-10")
META_PATH = RUN_DIR / "artifacts" / "vectors" / "meta.json"
EXPORT_DIR = RUN_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Load metadata
# -------------------------------------------------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

layer_accuracies = meta["layer_accuracies"]
best_layer = meta["best_layer"]
best_accuracy = meta["best_accuracy"]
experiment_name = meta.get("experiment_name", "LukeActsVec")
gospels = meta.get("gospels", [])
num_layers = meta.get("num_layers", len(layer_accuracies))


# -------------------------------------------------
# Print summary
# -------------------------------------------------
print("Experiment:", experiment_name)
print("Books:", gospels)
print("Best layer:", best_layer)
print("Best accuracy:", best_accuracy)
print("Number of layers:", num_layers)


# -------------------------------------------------
# Plot
# -------------------------------------------------
layers = list(range(len(layer_accuracies)))

plt.figure(figsize=(10, 6))
plt.plot(layers, layer_accuracies, marker="o")
plt.axvline(best_layer, linestyle="--", linewidth=1)
plt.axhline(best_accuracy, linestyle="--", linewidth=1)

plt.title(f"{experiment_name} — Layer Accuracy")
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.xticks(layers if len(layers) <= 40 else range(0, len(layers), 2))
plt.grid(True, alpha=0.3)

# Annotate best point
plt.annotate(
    f"Best layer = {best_layer}\nAccuracy = {best_accuracy:.4f}",
    xy=(best_layer, best_accuracy),
    xytext=(best_layer + 1, best_accuracy - 0.08),
    arrowprops=dict(arrowstyle="->"),
)

plt.tight_layout()

# -------------------------------------------------
# Save
# -------------------------------------------------
png_path = EXPORT_DIR / "LukeActsVec_Run1_Layer_Accuracy.png"
plt.savefig(png_path, dpi=300)
print(f"Saved plot to: {png_path}")

plt.show()