import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("data/raw/pill")

def run_eda():
    print("=" * 50)
    print("MVTec Pill Dataset — EDA")
    print("=" * 50)

    split_stats = {}

    for split in ["train", "test"]:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            print(f"[WARN] {split_dir} not found — skipping")
            continue

        class_counts = defaultdict(int)
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            count = len(list(class_dir.glob("*.png")))
            class_counts[class_name] = count

        split_stats[split] = class_counts
        total = sum(class_counts.values())

        print(f"\n{split.upper()} split — {total} images")
        print("-" * 30)
        for cls, cnt in sorted(class_counts.items()):
            tag = "  [GOOD]" if cls == "good" else "[DEFECT]"
            bar = "#" * cnt
            print(f"  {tag}  {cls:<20} {cnt:>4} images  {bar}")

    print("\n" + "=" * 50)
    good_train = split_stats.get("train", {}).get("good", 0)
    defect_test = sum(v for k, v in split_stats.get("test", {}).items() if k != "good")
    good_test   = split_stats.get("test", {}).get("good", 0)
    print(f"Train good images  : {good_train}")
    print(f"Test good images   : {good_test}")
    print(f"Test defect images : {defect_test}")
    print(f"Defect classes     : {list(split_stats.get('test', {}).keys())}")

    save_sample_grid()

def save_sample_grid():
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("MVTec Pill — sample images", fontsize=14)

    test_dir = DATA_ROOT / "test"
    classes  = sorted(os.listdir(test_dir)) if test_dir.exists() else []
    classes  = [c for c in classes if (test_dir / c).is_dir()][:8]

    for ax, cls in zip(axes.flatten(), classes):
        imgs = list((test_dir / cls).glob("*.png"))
        if imgs:
            img = mpimg.imread(str(imgs[0]))
            ax.imshow(img)
            tag = "GOOD" if cls == "good" else f"DEFECT: {cls}"
            ax.set_title(tag, fontsize=9,
                         color="green" if cls == "good" else "red")
        ax.axis("off")

    for ax in axes.flatten()[len(classes):]:
        ax.axis("off")

    out = Path("notebooks/sample_grid.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nSample grid saved → {out}")

if __name__ == "__main__":
    run_eda()