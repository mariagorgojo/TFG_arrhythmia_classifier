from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = PROJECT_ROOT / "data" / "raw_xml" / "example.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "examples" / "one_xml_extraction"
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
class ExtractedOccurrence:
    occurrence_id: str
    occurrence_index_in_xml: int
    source_xml_name: str
    source_xml_path: Path
    device_id: str
    device_model: str | None
    device_serial_number: str | None
    episode_id: str | None
    occurrence_type: str
    occurrence_datetime: str | None
    sample_interval_seconds: float | None
    sampling_rate_hz: float | None
    amplitude_scale_factor: float | None
    amplitude_unit: str | None
    sample_count: int
    waveform_duration_seconds: float | None
    segment_count: int
    has_stored_waveform: bool
    has_egm_not_stored: bool
    marker_count: int
    event_count: int
    waveform_path: Path | None
    markers_path: Path
    events_path: Path
    details_path: Path


def parse_iso8601_duration(duration_text: str | None) -> float | None:
    """Convert Medtronic ISO 8601 duration strings such as PT0.0078125S to seconds."""
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
    """Create a stable privacy-friendlier device identifier for dataset rows."""
    if not device_serial_number:
        return "device_unknown"
    digest = hashlib.sha256(device_serial_number.encode("utf-8")).hexdigest()
    return f"device_{digest[:12]}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_samples(segment: ET.Element) -> list[int]:
    raw_text = segment.attrib.get("samples", "")
    if not raw_text.strip():
        return []
    return [int(value) for value in raw_text.split()]


def extract_occurrences(xml_path: Path, output_dir: Path) -> list[ExtractedOccurrence]:
    """Extract all CardiacOccurrenceRecord entries from one XML file."""
    root = ET.parse(xml_path).getroot()

    device_serial_number = root.attrib.get("deviceSerialNumber")
    device_model = root.attrib.get("deviceModelId")
    device_id = anonymize_device_id(device_serial_number)

    xml_output_dir = output_dir / xml_path.stem
    waveform_dir = xml_output_dir / "waveforms"
    marker_dir = xml_output_dir / "markers"
    event_dir = xml_output_dir / "events"
    details_dir = xml_output_dir / "details"

    extracted: list[ExtractedOccurrence] = []
    occurrence_records = root.findall(".//CardiacOccurrenceRecord")

    for occurrence_index, occurrence_record in enumerate(occurrence_records, start=1):
        occurrence_id = f"{xml_path.stem}_occurrence_{occurrence_index:03d}"
        occurrence_type = text_at(occurrence_record, "./OccurrenceType/Discrete") or "Unknown"
        occurrence_datetime = text_at(occurrence_record, "./OccurrenceDateTime/DateTime")
        episode_id = text_at(occurrence_record, "./EpisodeID/Integer")

        episode_record = occurrence_record.find("./EpisodeRecord")
        waveform_channel = (
            episode_record.find(".//WaveformChannel") if episode_record is not None else None
        )

        sample_interval_seconds: float | None = None
        sampling_rate_hz: float | None = None
        amplitude_scale_factor: float | None = None
        amplitude_unit: str | None = None
        waveform_rows: list[dict[str, object]] = []
        segment_rows: list[dict[str, object]] = []

        if waveform_channel is not None:
            sample_interval_seconds = parse_iso8601_duration(
                waveform_channel.attrib.get("sampleInterval")
            )
            sampling_rate_hz = (
                1.0 / sample_interval_seconds
                if sample_interval_seconds and sample_interval_seconds > 0
                else None
            )
            amplitude_scale_factor = float_or_none(
                waveform_channel.attrib.get("amplitudeScaleFactor")
                or waveform_channel.attrib.get("amplitudeResolution")
            )
            amplitude_unit = waveform_channel.attrib.get("amplitudeUnit")

            running_sample_index = 0
            for segment_index, segment in enumerate(
                waveform_channel.findall("./WaveformSegment"), start=1
            ):
                state = segment.attrib.get("state", "Unknown")
                segment_offset_seconds = parse_iso8601_duration(segment.attrib.get("offset"))
                segment_length_seconds = parse_iso8601_duration(segment.attrib.get("length"))
                raw_samples = extract_samples(segment)

                segment_rows.append(
                    {
                        "segment_index": segment_index,
                        "state": state,
                        "offset_seconds": segment_offset_seconds,
                        "length_seconds": segment_length_seconds,
                        "sample_count": len(raw_samples),
                    }
                )

                for raw_value in raw_samples:
                    value_in_unit = (
                        raw_value * amplitude_scale_factor
                        if amplitude_scale_factor is not None
                        else raw_value
                    )
                    time_seconds = (
                        running_sample_index * sample_interval_seconds
                        if sample_interval_seconds is not None
                        else None
                    )
                    waveform_rows.append(
                        {
                            "sample_index": running_sample_index,
                            "time_seconds": time_seconds,
                            "raw_value": raw_value,
                            "value": value_in_unit,
                            "unit": amplitude_unit,
                        }
                    )
                    running_sample_index += 1

        waveform_path = waveform_dir / f"{occurrence_id}_waveform.csv"
        if waveform_rows:
            write_csv(
                waveform_path,
                ["sample_index", "time_seconds", "raw_value", "value", "unit"],
                waveform_rows,
            )
        else:
            waveform_path = None

        marker_rows: list[dict[str, object]] = []
        if episode_record is not None:
            for marker_index, marker in enumerate(
                episode_record.findall(".//IntervalMarkerSegment//Mkr"), start=1
            ):
                marker_rows.append(
                    {
                        "marker_index": marker_index,
                        "marker_name": marker.attrib.get("marker"),
                        "offset_seconds": parse_iso8601_duration(marker.attrib.get("offset")),
                        "vv_interval_seconds": parse_iso8601_duration(
                            text_at(marker, "./VV")
                        ),
                    }
                )

        markers_path = marker_dir / f"{occurrence_id}_markers.csv"
        write_csv(
            markers_path,
            ["marker_index", "marker_name", "offset_seconds", "vv_interval_seconds"],
            marker_rows,
        )

        event_rows: list[dict[str, object]] = []
        if episode_record is not None:
            for event_index, event in enumerate(
                episode_record.findall(".//EventSegment/Event"), start=1
            ):
                event_rows.append(
                    {
                        "event_index": event_index,
                        "event_id": event.attrib.get("id"),
                        "event_type": event.attrib.get("type"),
                        "offset_seconds": parse_iso8601_duration(event.attrib.get("offset")),
                    }
                )

        events_path = event_dir / f"{occurrence_id}_events.csv"
        write_csv(
            events_path,
            ["event_index", "event_id", "event_type", "offset_seconds"],
            event_rows,
        )

        details_path = details_dir / f"{occurrence_id}_details.json"
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details = {
            "occurrence_id": occurrence_id,
            "source_xml_name": xml_path.name,
            "source_xml_path": str(xml_path),
            "device_id": device_id,
            "device_serial_number": device_serial_number,
            "device_model": device_model,
            "episode_id": episode_id,
            "occurrence_index_in_xml": occurrence_index,
            "occurrence_type": occurrence_type,
            "occurrence_datetime": occurrence_datetime,
            "sample_interval_seconds": sample_interval_seconds,
            "sampling_rate_hz": sampling_rate_hz,
            "amplitude_scale_factor": amplitude_scale_factor,
            "amplitude_unit": amplitude_unit,
            "segments": segment_rows,
            "waveform_path": str(waveform_path) if waveform_path else None,
            "markers_path": str(markers_path),
            "events_path": str(events_path),
        }
        details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

        sample_count = len(waveform_rows)
        waveform_duration_seconds = (
            sample_count * sample_interval_seconds
            if sample_interval_seconds is not None
            else None
        )
        extracted.append(
            ExtractedOccurrence(
                occurrence_id=occurrence_id,
                occurrence_index_in_xml=occurrence_index,
                source_xml_name=xml_path.name,
                source_xml_path=xml_path,
                device_id=device_id,
                device_model=device_model,
                device_serial_number=device_serial_number,
                episode_id=episode_id,
                occurrence_type=occurrence_type,
                occurrence_datetime=occurrence_datetime,
                sample_interval_seconds=sample_interval_seconds,
                sampling_rate_hz=sampling_rate_hz,
                amplitude_scale_factor=amplitude_scale_factor,
                amplitude_unit=amplitude_unit,
                sample_count=sample_count,
                waveform_duration_seconds=waveform_duration_seconds,
                segment_count=len(segment_rows),
                has_stored_waveform=sample_count > 0,
                has_egm_not_stored=any(
                    row["state"] == "EgmNotStored" for row in segment_rows
                ),
                marker_count=len(marker_rows),
                event_count=len(event_rows),
                waveform_path=waveform_path,
                markers_path=markers_path,
                events_path=events_path,
                details_path=details_path,
            )
        )

    return extracted


def write_occurrences_summary(
    occurrences: list[ExtractedOccurrence], output_dir: Path, xml_path: Path
) -> Path:
    summary_path = output_dir / xml_path.stem / "occurrences.csv"
    rows = []
    for occurrence in occurrences:
        rows.append(
            {
                "occurrence_id": occurrence.occurrence_id,
                "source_xml_name": occurrence.source_xml_name,
                "source_xml_path": str(occurrence.source_xml_path),
                "device_id": occurrence.device_id,
                "device_model": occurrence.device_model,
                "episode_id": occurrence.episode_id,
                "occurrence_index_in_xml": occurrence.occurrence_index_in_xml,
                "occurrence_type": occurrence.occurrence_type,
                "occurrence_datetime": occurrence.occurrence_datetime,
                "sample_count": occurrence.sample_count,
                "sample_interval_seconds": occurrence.sample_interval_seconds,
                "sampling_rate_hz": occurrence.sampling_rate_hz,
                "waveform_duration_seconds": occurrence.waveform_duration_seconds,
                "amplitude_scale_factor": occurrence.amplitude_scale_factor,
                "amplitude_unit": occurrence.amplitude_unit,
                "segment_count": occurrence.segment_count,
                "has_stored_waveform": occurrence.has_stored_waveform,
                "has_egm_not_stored": occurrence.has_egm_not_stored,
                "marker_count": occurrence.marker_count,
                "event_count": occurrence.event_count,
                "waveform_path": str(occurrence.waveform_path)
                if occurrence.waveform_path
                else "",
                "markers_path": str(occurrence.markers_path),
                "events_path": str(occurrence.events_path),
                "details_path": str(occurrence.details_path),
            }
        )

    write_csv(
        summary_path,
        [
            "occurrence_id",
            "source_xml_name",
            "source_xml_path",
            "device_id",
            "device_model",
            "episode_id",
            "occurrence_index_in_xml",
            "occurrence_type",
            "occurrence_datetime",
            "sample_count",
            "sample_interval_seconds",
            "sampling_rate_hz",
            "waveform_duration_seconds",
            "amplitude_scale_factor",
            "amplitude_unit",
            "segment_count",
            "has_stored_waveform",
            "has_egm_not_stored",
            "marker_count",
            "event_count",
            "waveform_path",
            "markers_path",
            "events_path",
            "details_path",
        ],
        rows,
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract all CardiacOccurrenceRecord entries from one Medtronic XML."
    )
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=DEFAULT_XML_PATH,
        help=f"XML file to extract. Default: {DEFAULT_XML_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for extracted review files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    occurrences = extract_occurrences(args.xml_path, args.output_dir)
    summary_path = write_occurrences_summary(occurrences, args.output_dir, args.xml_path)

    print(f"XML file: {args.xml_path}")
    print(f"Output folder: {summary_path.parent}")
    print(f"Occurrences extracted: {len(occurrences)}")
    print(f"Summary CSV: {summary_path}")
    for occurrence in occurrences:
        print(
            " - "
            f"{occurrence.occurrence_id}: "
            f"label={occurrence.occurrence_type}, "
            f"episode_id={occurrence.episode_id}, "
            f"samples={occurrence.sample_count}, "
            f"sampling_rate_hz={occurrence.sampling_rate_hz}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
