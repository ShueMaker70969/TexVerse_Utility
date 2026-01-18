from pathlib import Path
import argparse
import sys

from download_model_by_id import (
    TexVerseDownloader,
    normalize_model_id,
)


def download_n_models(
    model_ids: list[str],
    *,
    config_path: Path,
    output_dir: Path,
    mode: str | None = None,
    resolution: int | None = None,
) -> None:
    """
    Download multiple TexVerse models efficiently.

    Metadata is loaded ONCE and reused.
    """
    downloader = TexVerseDownloader(config_path)

    # Override output directory programmatically
    downloader.output_dir = output_dir.resolve()
    downloader.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, raw_id in enumerate(model_ids, start=1):
        try:
            model_id = normalize_model_id(raw_id)
            print(f"[{idx}/{len(model_ids)}] Downloading {model_id} ...")

            path = downloader.download(
                model_id,
                mode=mode,
                resolution=resolution,
            )

            print(f"  -> saved to {path}")

        except Exception as exc:
            print(f"[warn] Failed to download {raw_id}: {exc}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download N TexVerse models using metadata-aware downloader."
    )

    parser.add_argument(
        "model_ids",
        nargs="+",
        help="List of TexVerse model IDs (or text containing them).",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml (default: ./config.yaml).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloaded_models"),
        help="Target directory for downloaded models.",
    )

    parser.add_argument(
        "--mode",
        choices=["highest_available", "fixed"],
        help="Download mode override.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        help="Resolution to use when mode=fixed (e.g. 1024).",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    download_n_models(
        model_ids=args.model_ids,
        config_path=args.config.resolve(),
        output_dir=args.output_dir,
        mode=args.mode,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
