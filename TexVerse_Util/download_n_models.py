from pathlib import Path
import argparse
import sys

from download_model_by_id import TexVerseDownloader, normalize_model_id


def load_model_ids_from_text(path: Path) -> list[str]:
    """
    Load all model IDs from a text file.
    """
    if not path.exists():
        raise FileNotFoundError(f"textdata file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def download_n_models_from_config(
    *,
    config_path: Path,
    n_models: int,
    output_dir: Path | None = None,
    mode: str | None = None,
    resolution: int | None = None,
) -> None:
    """
    Download N *new* models listed in textdata_path from config.yaml.
    Already-downloaded models are skipped.
    """
    downloader = TexVerseDownloader(config_path)

    # Resolve textdata path from config
    textdata_path_value = downloader.config.get("textdata_path")
    if not textdata_path_value:
        raise ValueError("textdata_path is not defined in config.yaml")

    textdata_path = downloader._resolve_path(textdata_path_value)
    model_ids = load_model_ids_from_text(textdata_path)

    if not model_ids:
        raise ValueError("No model IDs found in textdata file.")

    # Optional output directory override
    if output_dir is not None:
        downloader.output_dir = output_dir.resolve()
        downloader.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Target: {n_models} new models")
    print(f"[info] Output directory: {downloader.output_dir}")

    downloaded = 0
    scanned = 0

    for raw_id in model_ids:
        if downloaded >= n_models:
            break

        scanned += 1

        try:
            model_id = normalize_model_id(raw_id)

            # Predict final path WITHOUT downloading
            expected_path = downloader.output_dir / model_id

            if expected_path.exists():
                print(f"[skip] {model_id} (already exists)")
                continue

            print(f"[{downloaded + 1}/{n_models}] Downloading {model_id}")

            local_path = downloader.download(
                model_id,
                mode=mode,
                resolution=resolution,
            )

            print(f"  -> saved to {local_path}")
            downloaded += 1

        except Exception as exc:
            print(f"[warn] Failed {raw_id}: {exc}", file=sys.stderr)

    print(
        f"[done] Downloaded {downloaded} new models "
        f"(scanned {scanned} entries)"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download N new TexVerse models from textdata_path."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )

    parser.add_argument(
        "--n-models",
        type=int,
        required=True,
        help="Number of NEW models to download.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory.",
    )

    parser.add_argument(
        "--mode",
        choices=["highest_available", "fixed"],
        help="Override download mode.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        help="Resolution when mode=fixed (e.g. 1024).",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    download_n_models_from_config(
        config_path=args.config.resolve(),
        n_models=args.n_models,
        output_dir=args.output_dir,
        mode=args.mode,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
