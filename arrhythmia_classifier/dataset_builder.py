from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_DIR = PROJECT_ROOT / "data" / "pdds"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "occurrence_dataset"
_DURATION_RE = re.compile(
    r"^P"
    r"(?:(?P<days>\d+(?:\.\d+)?)D)?"
    r"(?:T"
    r"(?:(?P<hours>\d+(?:\.\d+)?)H)?"
    r"(?:(?P<minutes>\d+(?:\.\d+)?)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
    r")?$"
)


@dataclass(slots=True)
class OccurrenceDatasetRow:
    occurrence_id: str
    source_xml_name: str
    source_xml_path: str
    device_id: str
    device_model: str | None
    episode_id: str | None
    occurrence_index_in_xml: int
    occurrence_type: str
    occurrence_datetime: str | None
    sample_interval_seconds: float | None
    sampling_rate_hz: float | None
    amplitude_scale_factor: float | None
    amplitude_unit: str | None
    sample_count: int
    waveform_duration_seconds: float | None
    stored_segment_count: int
    total_segment_count: int
    marker_count: int
    event_count: int
    has_stored_waveform: bool
    has_egm_not_stored: bool
    waveform_npz_path: str
    details_json_path: str


def parse_iso8601_duration(duration_text: str | None) -> float | None:
    if not duration_text:
        return None
    match = _DURATION_RE.match(duration_text.strip())
    if not match:
        return None
    parts = {name: float(value or 0.0) for name, value in match.groupdict().items()}
    return (
        parts["days"] * 86400.0
        + parts["hours"] * 3600.0
        + parts["minutes"] * 60.0
        + parts["seconds"]
    )


def text_at(element: ET.Element, path: str) -> str | None:
    found = element.find(path)
    if found is None or found.text is None:
        return None
    value = found.text.strip()
    return value or None


def float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def anonymize_device_id(device_serial_number: str | None) -> str:
    if not device_serial_number:
        return "device_unknown"
    digest = hashlib.sha256(device_serial_number.encode("utf-8")).hexdigest()
    return f"device_{digest[:12]}"


def iter_xml_files(xml_dir: Path, *, limit: int | None = None) -> list[Path]:
    paths = sorted(path for path in xml_dir.glob("*.xml") if path.is_file())
    return paths[:limit] if limit is not None else paths


def _safe_occurrence_id(xml_path: Path, occurrence_index: int) -> str:
    return f"{xml_path.stem}_occurrence_{occurrence_index:03d}"


def _read_waveform(
    waveform_channel: ET.Element | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float | None,
    float | None,
    str | None,
    list[dict[str, object]],
]:
    if waveform_channel is None:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, np.asarray([], dtype=int), None, None, None, []

    sample_interval_seconds = parse_iso8601_duration(
        waveform_channel.attrib.get("sampleInterval")
    )
    amplitude_scale_factor = float_or_none(
        waveform_channel.attrib.get("amplitudeScaleFactor")
        or waveform_channel.attrib.get("amplitudeResolution")
    )
    amplitude_unit = waveform_channel.attrib.get("amplitudeUnit")

    raw_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    segment_index_chunks: list[np.ndarray] = []
    segments: list[dict[str, object]] = []
    fallback_sample_index = 0

    for segment_index, segment in enumerate(
        waveform_channel.findall("./WaveformSegment"), start=1
    ):
        state = segment.attrib.get("state", "Unknown")
        offset_seconds = parse_iso8601_duration(segment.attrib.get("offset"))
        length_seconds = parse_iso8601_duration(segment.attrib.get("length"))
        raw_text = segment.attrib.get("samples", "")
        raw_samples = np.fromstring(raw_text, sep=" ", dtype=np.float32)

        segments.append(
            {
                "segment_index": segment_index,
                "state": state,
                "offset_seconds": offset_seconds,
                "length_seconds": length_seconds,
                "sample_count": int(raw_samples.size),
            }
        )

        if raw_samples.size == 0:
            continue

        if amplitude_scale_factor is None:
            waveform_values = raw_samples.astype(np.float32, copy=True)
        else:
            waveform_values = raw_samples * np.float32(amplitude_scale_factor)

        if sample_interval_seconds is None:
            times = np.full(raw_samples.size, np.nan, dtype=np.float32)
        else:
            if offset_seconds is None:
                start_time = fallback_sample_index * sample_interval_seconds
            else:
                start_time = offset_seconds
            times = start_time + np.arange(raw_samples.size, dtype=np.float32) * np.float32(
                sample_interval_seconds
            )

        raw_chunks.append(raw_samples)
        value_chunks.append(waveform_values)
        time_chunks.append(times.astype(np.float32))
        segment_index_chunks.append(
            np.full(raw_samples.size, segment_index, dtype=np.int16)
        )
        fallback_sample_index += int(raw_samples.size)

    if not raw_chunks:
        empty = np.asarray([], dtype=float)
        return (
            empty,
            empty,
            empty,
            np.asarray([], dtype=int),
            sample_interval_seconds,
            amplitude_scale_factor,
            amplitude_unit,
            segments,
        )

    return (
        np.concatenate(raw_chunks),
        np.concatenate(value_chunks),
        np.concatenate(time_chunks),
        np.concatenate(segment_index_chunks),
        sample_interval_seconds,
        amplitude_scale_factor,
        amplitude_unit,
        segments,
    )


def _read_markers(episode_record: ET.Element | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if episode_record is None:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), []

    rr_intervals: list[float] = []
    marker_offsets: list[float] = []
    marker_names: list[str] = []
    for marker in episode_record.findall(".//IntervalMarkerSegment//Mkr"):
        marker_names.append(marker.attrib.get("marker", "Unknown"))
        offset = parse_iso8601_duration(marker.attrib.get("offset"))
        vv = parse_iso8601_duration(text_at(marker, "./VV"))
        marker_offsets.append(np.nan if offset is None else offset)
        rr_intervals.append(np.nan if vv is None else vv)

    return (
        np.asarray(rr_intervals, dtype=np.float32),
        np.asarray(marker_offsets, dtype=np.float32),
        marker_names,
    )


def _read_events(episode_record: ET.Element | None) -> list[dict[str, object]]:
    if episode_record is None:
        return []

    rows: list[dict[str, object]] = []
    for event_index, event in enumerate(
        episode_record.findall(".//EventSegment/Event"), start=1
    ):
        rows.append(
            {
                "event_index": event_index,
                "event_id": event.attrib.get("id"),
                "event_type": event.attrib.get("type"),
                "offset_seconds": parse_iso8601_duration(event.attrib.get("offset")),
            }
        )
    return rows


def extract_xml_to_dataset(xml_path: Path, output_dir: Path) -> list[OccurrenceDatasetRow]:
    root = ET.parse(xml_path).getroot()
    device_serial_number = root.attrib.get("deviceSerialNumber")
    device_id = anonymize_device_id(device_serial_number)
    device_model = root.attrib.get("deviceModelId")

    arrays_dir = output_dir / "arrays"
    details_dir = output_dir / "details"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    rows: list[OccurrenceDatasetRow] = []
    for occurrence_index, occurrence_record in enumerate(
        root.findall(".//CardiacOccurrenceRecord"), start=1
    ):
        occurrence_id = _safe_occurrence_id(xml_path, occurrence_index)
        occurrence_type = (
            text_at(occurrence_record, "./OccurrenceType/Discrete") or "Unknown"
        )
        episode_id = text_at(occurrence_record, "./EpisodeID/Integer")
        occurrence_datetime = text_at(
            occurrence_record, "./OccurrenceDateTime/DateTime"
        )
        episode_record = occurrence_record.find("./EpisodeRecord")
        waveform_channel = (
            episode_record.find(".//WaveformChannel")
            if episode_record is not None
            else None
        )

        (
            raw_samples,
            waveform_values,
            time_seconds,
            segment_indices,
            sample_interval_seconds,
            amplitude_scale_factor,
            amplitude_unit,
            segments,
        ) = _read_waveform(waveform_channel)
        rr_intervals, marker_offsets, marker_names = _read_markers(episode_record)
        events = _read_events(episode_record)

        waveform_npz_path = arrays_dir / f"{occurrence_id}.npz"
        np.savez_compressed(
            waveform_npz_path,
            raw_samples=raw_samples.astype(np.float32),
            waveform_values=waveform_values.astype(np.float32),
            time_seconds=time_seconds.astype(np.float32),
            segment_indices=segment_indices.astype(np.int16),
            rr_intervals_seconds=rr_intervals,
            marker_offsets_seconds=marker_offsets,
            marker_names=np.asarray(marker_names, dtype="U32"),
        )

        details_json_path = details_dir / f"{occurrence_id}.json"
        details = {
            "occurrence_id": occurrence_id,
            "source_xml_name": xml_path.name,
            "source_xml_path": str(xml_path),
            "device_id": device_id,
            "device_model": device_model,
            "episode_id": episode_id,
            "occurrence_index_in_xml": occurrence_index,
            "occurrence_type": occurrence_type,
            "occurrence_datetime": occurrence_datetime,
            "sample_interval_seconds": sample_interval_seconds,
            "sampling_rate_hz": (
                1.0 / sample_interval_seconds
                if sample_interval_seconds and sample_interval_seconds > 0
                else None
            ),
            "amplitude_scale_factor": amplitude_scale_factor,
            "amplitude_unit": amplitude_unit,
            "segments": segments,
            "events": events,
            "waveform_npz_path": str(waveform_npz_path),
        }
        details_json_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

        sample_count = int(waveform_values.size)
        waveform_duration_seconds = (
            sample_count * sample_interval_seconds
            if sample_interval_seconds is not None
            else None
        )
        rows.append(
            OccurrenceDatasetRow(
                occurrence_id=occurrence_id,
                source_xml_name=xml_path.name,
                source_xml_path=str(xml_path),
                device_id=device_id,
                device_model=device_model,
                episode_id=episode_id,
                occurrence_index_in_xml=occurrence_index,
                occurrence_type=occurrence_type,
                occurrence_datetime=occurrence_datetime,
                sample_interval_seconds=sample_interval_seconds,
                sampling_rate_hz=details["sampling_rate_hz"],
                amplitude_scale_factor=amplitude_scale_factor,
                amplitude_unit=amplitude_unit,
                sample_count=sample_count,
                waveform_duration_seconds=waveform_duration_seconds,
                stored_segment_count=sum(1 for segment in segments if segment["sample_count"]),
                total_segment_count=len(segments),
                marker_count=int(rr_intervals.size),
                event_count=len(events),
                has_stored_waveform=sample_count > 0,
                has_egm_not_stored=any(
                    segment["state"] == "EgmNotStored" for segment in segments
                ),
                waveform_npz_path=str(waveform_npz_path),
                details_json_path=str(details_json_path),
            )
        )

    return rows


def load_existing_dataset_row(
    details_json_path: Path, waveform_npz_path: Path
) -> OccurrenceDatasetRow | None:
    if not details_json_path.exists() or not waveform_npz_path.exists():
        return None

    details = json.loads(details_json_path.read_text(encoding="utf-8"))
    arrays = np.load(waveform_npz_path)
    waveform_values = arrays["waveform_values"]
    rr_intervals = arrays["rr_intervals_seconds"]
    segments = details.get("segments", [])
    events = details.get("events", [])
    sample_interval_seconds = details.get("sample_interval_seconds")
    sample_count = int(waveform_values.size)
    waveform_duration_seconds = (
        sample_count * sample_interval_seconds
        if sample_interval_seconds is not None
        else None
    )

    return OccurrenceDatasetRow(
        occurrence_id=details["occurrence_id"],
        source_xml_name=details["source_xml_name"],
        source_xml_path=details["source_xml_path"],
        device_id=details["device_id"],
        device_model=details.get("device_model"),
        episode_id=details.get("episode_id"),
        occurrence_index_in_xml=int(details["occurrence_index_in_xml"]),
        occurrence_type=details["occurrence_type"],
        occurrence_datetime=details.get("occurrence_datetime"),
        sample_interval_seconds=sample_interval_seconds,
        sampling_rate_hz=details.get("sampling_rate_hz"),
        amplitude_scale_factor=details.get("amplitude_scale_factor"),
        amplitude_unit=details.get("amplitude_unit"),
        sample_count=sample_count,
        waveform_duration_seconds=waveform_duration_seconds,
        stored_segment_count=sum(1 for segment in segments if segment["sample_count"]),
        total_segment_count=len(segments),
        marker_count=int(rr_intervals.size),
        event_count=len(events),
        has_stored_waveform=sample_count > 0,
        has_egm_not_stored=any(segment["state"] == "EgmNotStored" for segment in segments),
        waveform_npz_path=str(waveform_npz_path),
        details_json_path=str(details_json_path),
    )


def append_manifest_rows(
    rows: list[OccurrenceDatasetRow], manifest_path: Path, *, write_header: bool
) -> None:
    fieldnames = list(OccurrenceDatasetRow.__dataclass_fields__)
    mode = "w" if write_header else "a"
    with manifest_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_manifest(rows: list[OccurrenceDatasetRow], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(OccurrenceDatasetRow.__dataclass_fields__)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_dataset(
    xml_dir: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    resume: bool = False,
    progress_every: int = 1000,
) -> list[OccurrenceDatasetRow]:
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML folder does not exist: {xml_dir}")
    if not xml_dir.is_dir():
        raise NotADirectoryError(f"XML path is not a folder: {xml_dir}")

    xml_paths = iter_xml_files(xml_dir, limit=limit)
    rows: list[OccurrenceDatasetRow] = []
    manifest_path = output_dir / "occurrences_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    for xml_index, xml_path in enumerate(xml_paths, start=1):
        if resume:
            root = ET.parse(xml_path).getroot()
            occurrence_count = len(root.findall(".//CardiacOccurrenceRecord"))
            xml_rows: list[OccurrenceDatasetRow] = []
            for occurrence_index in range(1, occurrence_count + 1):
                occurrence_id = _safe_occurrence_id(xml_path, occurrence_index)
                existing_row = load_existing_dataset_row(
                    output_dir / "details" / f"{occurrence_id}.json",
                    output_dir / "arrays" / f"{occurrence_id}.npz",
                )
                if existing_row is None:
                    xml_rows = extract_xml_to_dataset(xml_path, output_dir)
                    break
                xml_rows.append(existing_row)
        else:
            xml_rows = extract_xml_to_dataset(xml_path, output_dir)

        append_manifest_rows(xml_rows, manifest_path, write_header=not wrote_header)
        wrote_header = True
        rows.extend(xml_rows)

        if progress_every and xml_index % progress_every == 0:
            print(
                f"Processed {xml_index}/{len(xml_paths)} XML files; "
                f"{len(rows)} occurrences.",
                flush=True,
                file=sys.stderr,
            )

    if not wrote_header:
        write_manifest([], manifest_path)

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an occurrence-level dataset from Medtronic XML files."
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=DEFAULT_XML_DIR,
        help=f"Folder containing XML files. Default: {DEFAULT_XML_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output dataset folder. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of XML files to process for a quick trial run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing array/detail files when present and rebuild the manifest.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many XML files. Use 0 to disable.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = build_dataset(
        args.xml_dir,
        args.output_dir,
        limit=args.limit,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row.occurrence_type] = label_counts.get(row.occurrence_type, 0) + 1

    print(f"XML folder: {args.xml_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Occurrences extracted: {len(rows)}")
    print(f"Manifest: {args.output_dir / 'occurrences_manifest.csv'}")
    print("Labels:")
    for label, count in sorted(label_counts.items()):
        print(f" - {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
