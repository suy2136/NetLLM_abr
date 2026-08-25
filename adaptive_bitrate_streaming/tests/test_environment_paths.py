from pathlib import Path
import sys
import tempfile
import unittest


ABR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ABR_ROOT))

from baseline_special.env import Environment


class EnvironmentPathTest(unittest.TestCase):
    def test_video_size_directory_does_not_require_trailing_separator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = Path(temp_dir) / "video1_sizes"
            video_dir.mkdir()
            for bitrate in range(6):
                (video_dir / f"video_size_{bitrate}").write_text(
                    "1000\n", encoding="utf-8"
                )
            environment = Environment(
                all_cooked_time=[[0.0, 1.0]],
                all_cooked_bw=[[1.0, 1.0]],
                all_file_names=["trace"],
                all_mahimahi_ptrs=[1],
                video_size_dir=str(video_dir),
                fixed=True,
                trace_num=1,
            )
            self.assertEqual(environment.video_size[0], [1000])


if __name__ == "__main__":
    unittest.main()
