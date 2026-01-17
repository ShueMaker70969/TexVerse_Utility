"""
Download a TexVerse model by ID with configurable resolution preferences.

The module can be imported into other workflows (see `download_model_by_id`
function) or executed directly via CLI. Configuration is supplied through a
YAML file (defaults to `config.yaml`) that defines repositories, bucket
layouts, and default download mode.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PyYAML is required to parse config.yaml. Install it via `pip install pyyaml`."
    ) from exc

ID_PATTERN = re.compile(r"[0-9a-f]{32}", re.IGNORECASE)
RESOLUTION_PATTERN = re.compile(r"_([0-9]{3,5})\.glb$", re.IGNORECASE)


@dataclass(frozen=True)
class StorageLayout:
    """Holds repository/bucket information for a given resolution variant."""

    name: str
    repo_id: str
    repo_type: str
    base_dir: str
    filename_suffix: Optional[str] = None
    bucket_format: Optional[str] = None
    bucket_count: Optional[int] = None

    @property
    def normalized_base_dir(self) -> str:
        return self.base_dir.strip("/ ")

    def supports_bucket_scan(self) -> bool:
        return (
            self.filename_suffix is not None
            and self.bucket_format is not None
            and self.bucket_count is not None
        )


def normalize_model_id(raw_value: str) -> str:
    """Extract the 32-character TexVerse ID from arbitrary text."""
    match = ID_PATTERN.search(raw_value.strip())
    if not match:
        raise ValueError(f"Could not find a 32-character hex ID in '{raw_value}'.")
    return match.group(0).lower()


def parse_resolution_from_path(repo_path: str) -> Optional[int]:
    match = RESOLUTION_PATTERN.search(repo_path)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


class TexVerseDownloader:
    """
    Core downloader that coordinates metadata lookups and bucket scanning.

    Instantiate once and reuse if you're downloading many models in sequence,
    which avoids reloading `metadata.json` repeatedly.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent
        self.config = self._load_yaml(self.config_path)

        self.mode = self.config.get("mode", "highest_available")
        self.fixed_resolution = self.config.get("fixed_resolution")
        self.output_dir = self._resolve_path(self.config.get("output_dir", "downloaded_models"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.layouts = self._load_layouts(self.config.get("layouts", {}))
        self.base_dir_map = {layout.normalized_base_dir: layout for layout in self.layouts.values()}

        self.resolution_layouts = self._load_resolution_map(self.config.get("resolution_layouts", {}))
        fallback_name = self.config.get("fallback_bucket_layout")
        self.fallback_layout = self.layouts.get(fallback_name) if fallback_name else None

        metadata_path_value = self.config.get("metadata_path")
        self.metadata_index: Optional[Dict[str, dict]] = None
        if metadata_path_value:
            metadata_path = self._resolve_path(metadata_path_value)
            if metadata_path.exists():
                self.metadata_index = self._load_metadata(metadata_path)
            else:
                print(
                    f"[warn] metadata file not found at {metadata_path}. "
                    "Metadata-assisted resolution selection will be skipped.",
                    file=sys.stderr,
                )

    def _resolve_path(self, value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (self.config_dir / candidate).resolve()
        return candidate

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _load_metadata(path: Path) -> Dict[str, dict]:
        print(f"[info] Loading metadata index from {path} ...")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        print(f"[info] Loaded metadata entries: {len(data):,}")
        return data

    def _load_layouts(self, raw_layouts: dict) -> Dict[str, StorageLayout]:
        layouts: Dict[str, StorageLayout] = {}
        for name, payload in raw_layouts.items():
            if not isinstance(payload, dict):
                continue
            layout = StorageLayout(
                name=name,
                repo_id=payload["repo_id"],
                repo_type=payload.get("repo_type", "dataset"),
                base_dir=payload["base_dir"],
                filename_suffix=payload.get("filename_suffix"),
                bucket_format=payload.get("bucket_format"),
                bucket_count=payload.get("bucket_count"),
            )
            layouts[name] = layout
        if not layouts:
            raise ValueError("No storage layouts defined in config.yaml.")
        return layouts

    def _load_resolution_map(self, mapping: dict) -> Dict[int, StorageLayout]:
        resolution_map: Dict[int, StorageLayout] = {}
        for key, layout_name in mapping.items():
            try:
                resolution = int(key)
            except (TypeError, ValueError):
                continue
            layout = self.layouts.get(layout_name)
            if layout:
                resolution_map[resolution] = layout
        return resolution_map

    def download(
        self,
        raw_model_id: str,
        *,
        mode: Optional[str] = None,
        resolution: Optional[int] = None,
    ) -> Path:
        """
        Download the requested model using the configured strategy.

        Args:
            raw_model_id: Raw ID or URL snippet containing the TexVerse ID.
            mode: Override for download mode (`highest_available` or `fixed`).
            resolution: Desired resolution (required when `mode="fixed"`).
        """
        model_id = normalize_model_id(raw_model_id)

        selected_mode = (mode or self.mode).lower()
        if selected_mode not in {"highest_available", "fixed"}:
            raise ValueError(f"Unsupported mode '{selected_mode}'.")

        target_resolution = resolution or self.fixed_resolution if selected_mode == "fixed" else None
        if selected_mode == "fixed" and not target_resolution:
            raise ValueError("Fixed mode requires a resolution (e.g., 1024).")
        if isinstance(target_resolution, str):
            target_resolution = int(target_resolution)

        # 1) Metadata-assisted lookup.
        repo_path, layout = self._select_path_via_metadata(
            model_id, selected_mode, target_resolution
        )
        if repo_path and layout:
            return self._download_direct(layout, repo_path, model_id)

        # 2) Fallback bucket scanning.
        fallback_layout = (
            self._layout_for_resolution(target_resolution) if target_resolution else self.fallback_layout
        )
        if fallback_layout and fallback_layout.supports_bucket_scan():
            return self._download_via_bucket_scan(fallback_layout, model_id)

        raise FileNotFoundError(
            f"Could not determine download path for model {model_id}. "
            "Ensure metadata.json is available or configure a bucket-enabled layout."
        )

    def _select_path_via_metadata(
        self,
        model_id: str,
        mode: str,
        target_resolution: Optional[int],
    ) -> Tuple[Optional[str], Optional[StorageLayout]]:
        if not self.metadata_index:
            return None, None
        entry = self.metadata_index.get(model_id)
        if not entry:
            return None, None
        glb_paths: Iterable[str] = entry.get("glb_paths", [])
        candidates: List[Tuple[int, str, StorageLayout]] = []
        for repo_path in glb_paths:
            layout = self._layout_for_repo_path(repo_path)
            if not layout:
                continue
            resolution = parse_resolution_from_path(repo_path) or 0
            candidates.append((resolution, repo_path, layout))
        if not candidates:
            return None, None

        if mode == "fixed" and target_resolution:
            for res, path, layout in candidates:
                if res == target_resolution:
                    return path, layout
            return None, None

        # Highest available, prefer the highest resolution value.
        res, path, layout = max(candidates, key=lambda item: (item[0], item[1]))
        if res == 0 and mode == "highest_available":
            # No numeric resolution metadata, still return the first option.
            return path, layout
        return path, layout

    def _layout_for_repo_path(self, repo_path: str) -> Optional[StorageLayout]:
        normalized = repo_path.strip("/ ")
        for base_dir, layout in self.base_dir_map.items():
            prefix = f"{base_dir}/"
            if normalized.startswith(prefix):
                return layout
        return None

    def _layout_for_resolution(self, resolution: Optional[int]) -> Optional[StorageLayout]:
        if resolution is None:
            return None
        return self.resolution_layouts.get(int(resolution))

    def _download_direct(self, layout: StorageLayout, repo_path: str, model_id: str) -> Path:
        print(f"[info] Downloading {model_id} via metadata path {repo_path} ({layout.repo_id})")
        local_path = Path(
            hf_hub_download(
                repo_id=layout.repo_id,
                repo_type=layout.repo_type,
                filename=repo_path,
                local_dir=str(self.output_dir),
                local_dir_use_symlinks=False,
            )
        )
        return local_path

    def _download_via_bucket_scan(self, layout: StorageLayout, model_id: str) -> Path:
        if not layout.supports_bucket_scan():
            raise ValueError(f"Layout {layout.name} does not support bucket scanning.")

        last_error: Optional[Exception] = None
        for index in range(layout.bucket_count or 0):
            bucket = layout.bucket_format.format(index=index)
            repo_path = f"{layout.normalized_base_dir}/{bucket}/{model_id}{layout.filename_suffix}"
            try:
                return self._download_direct(layout, repo_path, model_id)
            except HfHubHTTPError as err:
                if err.response is not None and err.response.status_code == 404:
                    last_error = err
                    continue
                raise
        raise FileNotFoundError(
            f"Model {model_id} not found in configured buckets ({layout.name})."
        ) from last_error


@lru_cache(maxsize=None)
def _downloader_from_config(config_path: str) -> TexVerseDownloader:
    return TexVerseDownloader(Path(config_path))


def download_model_by_id(
    model_id: str,
    *,
    config_path: Path | str = Path("config.yaml"),
    mode: Optional[str] = None,
    resolution: Optional[int] = None,
) -> Path:
    """
    Convenience function for importing this module elsewhere.

    Example:
        from download_model_by_id import download_model_by_id
        path = download_model_by_id("d14eb14d83bd4e7ba7cbe443d76a10fd")
    """
    downloader = _downloader_from_config(str(Path(config_path).resolve()))
    return downloader.download(model_id, mode=mode, resolution=resolution)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a single TexVerse model by ID using config.yaml."
    )
    parser.add_argument("model_id", help="TexVerse model ID or text containing it.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=["highest_available", "fixed"],
        help="Override the configured download mode.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Resolution (e.g., 1024) used when mode=fixed.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        local_path = download_model_by_id(
            args.model_id,
            config_path=args.config,
            mode=args.mode,
            resolution=args.resolution,
        )
        print(f"[ok] Downloaded to {local_path}")
    except (ValueError, FileNotFoundError, HfHubHTTPError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
