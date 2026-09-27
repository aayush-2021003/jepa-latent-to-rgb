"""CPU-only checks for the walking/driving DenseWorld selection."""
from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from experiments.jepa_cosmos.download_denseworld import scan_tar


class TourTypeFilterTest(unittest.TestCase):
    def test_only_walking_and_driving_are_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "sample.tar"
            with tarfile.open(archive_path, "w") as archive:
                for index, tour_type in enumerate(("walking", "drive", "drone", "rain")):
                    base = f"{index:06d}"
                    metadata = {
                        "section": f"tier1/city/{tour_type}",
                        "video_id": f"source{index:05d}",
                        "source_file": f"{base}.mp4",
                        "tour_type": tour_type,
                    }
                    for extension, payload in (
                        ("json", json.dumps(metadata).encode()), ("mp4", b"video")
                    ):
                        member = tarfile.TarInfo(f"{base}.{extension}")
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
            records = scan_tar("remote.tar", archive_path, {"walking", "drive"})
            self.assertEqual({row.tour_type for row in records}, {"walking", "drive"})
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
