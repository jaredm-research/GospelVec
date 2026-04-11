import json
from pathlib import Path

import torch


# -----------------------------
# File paths
# -----------------------------
GOSPEL_BEST_PATH = Path(r"C:\Users\theKI\Documents\GospelVec\vectors\gospel_vectors_best.pt")
GOSPEL_META_PATH = Path(r"C:\Users\theKI\Documents\GospelVec\vectors\meta.json")

LUKEACTS_BEST_PATH = Path(r"C:\Users\theKI\Desktop\LukeActsVec_Run1_2026-04-10\vectors\lukeacts_vectors_best.pt")
LUKEACTS_META_PATH = Path(r"C:\Users\theKI\Desktop\LukeActsVec_Run1_2026-04-10\vectors\meta.json")


def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp(min=1e-8)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.dot(normalize(a), normalize(b)).item()


def main() -> None:
    # Load metadata
    gospel_meta = load_meta(GOSPEL_META_PATH)
    lukeacts_meta = load_meta(LUKEACTS_META_PATH)

    gospel_names = gospel_meta["gospels"]
    lukeacts_names = lukeacts_meta["gospels"]

    print("Original GospelVec order:", gospel_names)
    print("LukeActsVec order:", lukeacts_names)
    print()

    # Load tensors
    gospel_best = torch.load(GOSPEL_BEST_PATH, weights_only=True)
    lukeacts_best = torch.load(LUKEACTS_BEST_PATH, weights_only=True)

    print("gospel_vectors_best.pt shape:", tuple(gospel_best.shape))
    print("lukeacts_vectors_best.pt shape:", tuple(lukeacts_best.shape))
    print()

    # Find John and Acts indices from metadata
    try:
        john_idx = gospel_names.index("john")
    except ValueError:
        raise ValueError("Could not find 'john' in original GospelVec meta.json gospels list.")

    try:
        acts_idx = lukeacts_names.index("acts")
    except ValueError:
        raise ValueError("Could not find 'acts' in LukeActsVec meta.json gospels list.")

    john_vec = gospel_best[john_idx]
    acts_vec = lukeacts_best[acts_idx]

    # Direct John vs Acts
    john_acts = cosine(john_vec, acts_vec)
    print(f"John ↔ Acts cosine: {john_acts:+.6f}")
    print()

    # Acts vs each original GospelVec book
    print("Acts (LukeActsVec) vs original GospelVec books")
    print("-" * 48)
    for i, name in enumerate(gospel_names):
        sim = cosine(gospel_best[i], acts_vec)
        print(f"{name:>8} ↔ acts : {sim:+.6f}")
    print()

    # John vs each LukeActsVec book
    print("John (original GospelVec) vs LukeActsVec books")
    print("-" * 48)
    for i, name in enumerate(lukeacts_names):
        sim = cosine(john_vec, lukeacts_best[i])
        print(f"{'john':>8} ↔ {name:<8}: {sim:+.6f}")
    print()

    # Full cross-matrix, optional but useful
    print("Full cross-matrix: original GospelVec vs LukeActsVec")
    print("-" * 72)
    header = " " * 12 + "".join(f"{name:>12}" for name in lukeacts_names)
    print(header)
    for i, g_name in enumerate(gospel_names):
        row = [f"{g_name:>12}"]
        for j, la_name in enumerate(lukeacts_names):
            sim = cosine(gospel_best[i], lukeacts_best[j])
            row.append(f"{sim:>12.6f}")
        print("".join(row))


if __name__ == "__main__":
    main()