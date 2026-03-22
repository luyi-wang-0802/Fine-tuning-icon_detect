import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def _sum_metrics(items, key):
    # key in {"before","after"}
    tp = sum(x[key]["tp"] for x in items)
    fp = sum(x[key]["fp"] for x in items)
    fn = sum(x[key]["fn"] for x in items)
    pred = sum(x[key]["pred"] for x in items)
    gt = sum(x["gt"] for x in items)
    n = len(items)
    return {
        "tp": tp, "fp": fp, "fn": fn, "pred": pred, "gt": gt, "n": n,
        "pred_per_img": pred / n if n else 0.0,
        "gt_per_img": gt / n if n else 0.0,
        "fp_per_img": fp / n if n else 0.0,
        "fn_per_img": fn / n if n else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }

def make_dashboard(compare_dir="vw_compare_vis", out_name="dashboard.png", topk_imgs=6):
    compare_dir = Path(compare_dir)
    report_path = compare_dir / "compare_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = report["all"]  # sorted by score desc in earlier script
    top_items = report["top_improvements"][:topk_imgs]

    before = _sum_metrics(items, "before")
    after  = _sum_metrics(items, "after")

    # arrays for per-image distributions
    scores = np.array([x["score"] for x in items], dtype=float)
    b_fp = np.array([x["before"]["fp"] for x in items], dtype=float)
    b_fn = np.array([x["before"]["fn"] for x in items], dtype=float)
    a_fp = np.array([x["after"]["fp"] for x in items], dtype=float)
    a_fn = np.array([x["after"]["fn"] for x in items], dtype=float)

    # ---- Figure layout ----
    # 2 rows: top = charts, bottom = example images grid
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0])

    # (1) Overall TP/FP/FN bar
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["TP", "FP", "FN"]
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width/2, [before["tp"], before["fp"], before["fn"]], width, label="Before")
    ax1.bar(x + width/2, [after["tp"],  after["fp"],  after["fn"]],  width, label="After")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Overall counts (match IoU stats from report)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (2) FP/img & FN/img + precision/recall text
    ax2 = fig.add_subplot(gs[0, 1])
    labels2 = ["FP/img", "FN/img", "Pred/img"]
    x2 = np.arange(len(labels2))
    ax2.bar(x2 - width/2, [before["fp_per_img"], before["fn_per_img"], before["pred_per_img"]], width, label="Before")
    ax2.bar(x2 + width/2, [after["fp_per_img"],  after["fn_per_img"],  after["pred_per_img"]],  width, label="After")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2)
    ax2.set_title("Per-image operational view")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    txt = (
        f"Before: P={before['precision']:.3f} R={before['recall']:.3f}\n"
        f"After : P={after['precision']:.3f} R={after['recall']:.3f}\n"
        f"N images = {before['n']}"
    )
    ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, va="top")

    # (3) Score distribution + “who wins” summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(scores, bins=30)
    ax3.set_title("Per-image improvement score distribution")
    ax3.set_xlabel("score (higher = After better)")
    ax3.set_ylabel("count")
    ax3.grid(True, alpha=0.3)

    wins = (scores > 0).sum()
    ties = (scores == 0).sum()
    loses = (scores < 0).sum()
    ax3.text(
        0.02, 0.98,
        f"After better: {wins}\nTie: {ties}\nAfter worse: {loses}",
        transform=ax3.transAxes, va="top"
    )

    # ---- Bottom: example triplets ----
    # Grid for images inside gs[1, :] (one big cell then subgridspec)
    sub = gs[1, :].subgridspec(2, 3, wspace=0.02, hspace=0.08)

    for i in range(min(topk_imgs, 6)):
        ax = fig.add_subplot(sub[i // 3, i % 3])
        triplet_path = Path(top_items[i]["triplet"])
        if triplet_path.exists():
            img = mpimg.imread(str(triplet_path))
            ax.imshow(img)
            ax.set_title(f"Top #{i+1} score={top_items[i]['score']:.1f}")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{triplet_path}", ha="center", va="center")
        ax.axis("off")

    # Title includes main settings if present
    s = report.get("settings", {})
    title = (
        "Vectorworks Test Set: Before vs After (Visual Comparison Dashboard)\n"
        f"conf={s.get('conf')}  nms_iou={s.get('nms_iou')}  max_det={s.get('max_det')}  "
        f"match_iou_stats={s.get('match_iou_for_stats')}"
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = compare_dir / out_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved dashboard: {out_path}")

if __name__ == "__main__":
    make_dashboard(compare_dir="vw_compare_vis", out_name="dashboard.png", topk_imgs=6)
