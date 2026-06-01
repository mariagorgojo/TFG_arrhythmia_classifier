from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "pdds"
DEFAULT_DESTINATION_DIR = PROJECT_ROOT / "data" / "raw_xml"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "raw_xml_manifest.csv"


@dataclass(slots=True)
class CopyResult:
    source_path: Path
    destination_path: Path
    status: str
    size_bytes: int


def iter_xml_files(source_dir: Path) -> list[Path]:
    """Return all XML files below source_dir in stable order."""
    return sorted(path for path in source_dir.rglob("*.xml") if path.is_file())


def copy_xml_files(
    source_dir: Path,
    destination_dir: Path,
    *,
    overwrite: bool = False,
) -> list[CopyResult]:
    """Copy XML files into destination_dir without modifying the originals."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a folder: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    results: list[CopyResult] = []
    used_names: set[str] = set()

    for source_path in iter_xml_files(source_dir):
        destination_name = source_path.name
        destination_path = destination_dir / destination_name

        if destination_name in used_names:
            relative_parent = source_path.parent.relative_to(source_dir)
            safe_parent = "_".join(relative_parent.parts)
            destination_name = f"{safe_parent}_{source_path.name}" if safe_parent else source_path.name
            destination_path = destination_dir / destination_name

        used_names.add(destination_name)

        if destination_path.exists() and not overwrite:
            status = "skipped_exists"
        else:
            shutil.copy2(source_path, destination_path)
            status = "copied"

        results.append(
            CopyResult(
                source_path=source_path,
                destination_path=destination_path,
                status=status,
                size_bytes=source_path.stat().st_size,
            )
        )

    return results


def write_manifest(results: list[CopyResult], manifest_path: Path) -> None:
    """Write a manifest linking original XML paths to copied project paths."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "destination_path",
                "status",
                "size_bytes",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "source_path": str(result.source_path),
                    "destination_path": str(result.destination_path),
                    "status": result.status,
                    "size_bytes": result.size_bytes,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy Medtronic XML files into the project raw-data folder."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Folder containing XML files. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=DEFAULT_DESTINATION_DIR,
        help=f"Project folder where XML copies will be stored. Default: {DEFAULT_DESTINATION_DIR}",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"CSV manifest path. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing XML files in the destination folder.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    results = copy_xml_files(
        source_dir=args.source_dir,
        destination_dir=args.destination_dir,
        overwrite=args.overwrite,
    )
    write_manifest(results, args.manifest_path)

    copied = sum(1 for result in results if result.status == "copied")
    skipped = sum(1 for result in results if result.status == "skipped_exists")

    print(f"Source folder: {args.source_dir}")
    print(f"Destination folder: {args.destination_dir}")
    print(f"Manifest: {args.manifest_path}")
    print(f"XML files found: {len(results)}")
    print(f"Copied: {copied}")
    print(f"Skipped existing: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
