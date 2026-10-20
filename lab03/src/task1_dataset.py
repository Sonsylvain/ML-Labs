"""Task 1: Fashion-MNIST, normalised, 55k/5k/10k split, batch 128."""

from common import BATCH_SIZE, RESULTS_DIR, get_loaders


def main() -> None:
    train_loader, val_loader, test_loader = get_loaders()
    x, y = next(iter(train_loader))
    lines = [
        f"train: {len(train_loader.dataset)} samples, {len(train_loader)} batches",
        f"val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches",
        f"test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches",
        f"batch size {BATCH_SIZE}: x {tuple(x.shape)} {x.dtype}, y {tuple(y.shape)} {y.dtype}",
        f"first batch after Normalize: mean {x.mean():.4f}, std {x.std():.4f}",
    ]
    print("\n".join(lines))
    (RESULTS_DIR / "dataset.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
