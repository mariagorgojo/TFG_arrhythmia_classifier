from pathlib import Path

import numpy as np

from arrhythmia_classifier.dataset_builder import build_dataset, extract_xml_to_dataset


def _write_synthetic_xml(path: Path) -> None:
    samples = " ".join(str(value) for value in ([0, 1, -1, 2, -2] * 410)[:2048])
    markers = "".join(
        f'<Mkr marker="VS" offset="PT{index * 0.8}S"><VV>PT0.8S</VV></Mkr>'
        for index in range(11)
    )
    path.write_text(
        f"""<DeviceData deviceSerialNumber="TEST-DEVICE" deviceModelId="SyntheticModel">
  <CardiacOccurrenceRecord>
    <OccurrenceType><Discrete>CurrentECG</Discrete></OccurrenceType>
    <EpisodeID><Integer>1</Integer></EpisodeID>
    <OccurrenceDateTime><DateTime>2026-01-01T00:00:00</DateTime></OccurrenceDateTime>
    <EpisodeRecord>
      <WaveformChannel sampleInterval="PT0.0078125S" amplitudeScaleFactor="0.000815" amplitudeUnit="mV">
        <WaveformSegment state="Stored" offset="PT0S" length="PT16S" samples="{samples}" />
      </WaveformChannel>
      <IntervalMarkerSegment>{markers}</IntervalMarkerSegment>
    </EpisodeRecord>
  </CardiacOccurrenceRecord>
</DeviceData>
""",
        encoding="utf-8",
    )


def test_extract_xml_to_dataset_preserves_waveform_sampling(tmp_path: Path) -> None:
    fixture_xml = tmp_path / "synthetic.xml"
    _write_synthetic_xml(fixture_xml)
    rows = extract_xml_to_dataset(fixture_xml, tmp_path / "dataset")

    assert len(rows) == 1
    row = rows[0]
    assert row.occurrence_type == "CurrentECG"
    assert row.sample_interval_seconds == 0.0078125
    assert row.sampling_rate_hz == 128.0
    assert row.amplitude_scale_factor == 0.000815
    assert row.amplitude_unit == "mV"
    assert row.sample_count > 1000
    assert row.marker_count == 11

    arrays = np.load(row.waveform_npz_path)
    assert arrays["raw_samples"].shape == arrays["waveform_values"].shape
    assert arrays["waveform_values"].shape == arrays["time_seconds"].shape
    assert arrays["rr_intervals_seconds"].shape == (11,)
    assert arrays["time_seconds"][1] - arrays["time_seconds"][0] == np.float32(0.0078125)


def test_build_dataset_writes_manifest(tmp_path: Path) -> None:
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    _write_synthetic_xml(xml_dir / "synthetic.xml")

    rows = build_dataset(xml_dir, tmp_path / "dataset")

    assert len(rows) == 1
    manifest_path = tmp_path / "dataset" / "occurrences_manifest.csv"
    assert manifest_path.exists()
    assert "sample_interval_seconds" in manifest_path.read_text(encoding="utf-8")
