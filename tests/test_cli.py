import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mlarray import MLArray
from mlarray.cli import cli_convert_to_mlarray, cli_print_header, header


def _save(path):
    MLArray(
        np.zeros((2, 3, 4), dtype=np.float32),
        spacing=(1, 2, 3),
        space="patient",
        values={"kind": "measurement", "fill": float("nan")},
        profiles={"ml": {"statistics": {"min": float("-inf"), "max": float("inf")}}},
        user={"note": "x", "raw": b"\x00\x01\x02"},
        source=[{"format": "nifti", "header": {"descrip": "scan", "extension": b"\x00" * 16}}],
    ).save(path)


def _run(argv):
    out = io.StringIO()
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
        cli_print_header()
    return out.getvalue()


class TestHeader(unittest.TestCase):
    def test_prints_strict_json_without_source_and_user_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a.mla"
            _save(path)

            printed = json.loads(_run(["mlarray_header", str(path)]), parse_constant=self.fail)

            self.assertEqual(printed["version"], "1.0")
            self.assertEqual(printed["values"]["fill"], "NaN")
            self.assertEqual(printed["profiles"]["ml"]["statistics"], {"min": "-Infinity", "max": "Infinity"})
            self.assertNotIn("source", printed)
            self.assertNotIn("user", printed)

    def test_source_and_user_on_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a.mla"
            _save(path)

            printed = json.loads(_run(["mlarray_header", str(path), "--source", "--user"]))

            self.assertEqual(printed["source"][0]["header"]["extension"], "<binary, 16 bytes>")
            self.assertEqual(printed["user"], {"note": "x", "raw": "<binary, 3 bytes>"})

    def test_header_does_not_decode_source_unless_asked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a.mla"
            _save(path)
            with mock.patch("mlarray.metadata_io._decode_source", side_effect=AssertionError("decoded")):
                header(path)


class TestConvert(unittest.TestCase):
    def test_convert_explains_that_it_is_unavailable(self):
        with mock.patch.object(sys, "argv", ["mlarray_convert", "a.nii.gz", "b.mla"]):
            with self.assertRaisesRegex(RuntimeError, "MedVol 2"):
                cli_convert_to_mlarray()


if __name__ == "__main__":
    unittest.main()
