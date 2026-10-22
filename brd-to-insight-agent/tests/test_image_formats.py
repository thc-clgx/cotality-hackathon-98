import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import plotly.graph_objects as go

from brd_insight_agent.export import ExportError, figures_to_images


class ImageFormatsTests(unittest.TestCase):
    def worker(self, command, **kwargs):
        path = Path(command[-1])
        payload = json.loads(path.read_text())
        fmt = payload["format"]
        destination = path.parent / f"01.{fmt}"
        if fmt == "svg":
            destination.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><path d="M0 0 L10 10"/></svg>')
        else:
            with Image.new("RGB", (640, 480), "white") as image:
                image.save(destination, format=fmt.upper())
        return Mock(wait=Mock(return_value=0))

    def test_all_formats_share_timed_worker(self):
        for fmt in ("png", "svg", "jpeg"):
            with self.subTest(format=fmt), patch("brd_insight_agent.export.subprocess.Popen", side_effect=self.worker) as worker:
                images = figures_to_images([("test", go.Figure(go.Bar(x=[1], y=[2])))], fmt, 640, 480, 1)
                self.assertEqual(images[0][0], f"01_test.{fmt}")
                self.assertGreater(len(images[0][1]), 50)
                worker.assert_called_once()

    def test_invalid_format_rejected_before_process(self):
        with patch("brd_insight_agent.export.subprocess.Popen") as worker:
            with self.assertRaises(ExportError):
                figures_to_images([], "exe")
            worker.assert_not_called()