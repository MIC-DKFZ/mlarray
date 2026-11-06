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
from mlarray.cli import cli_convert_to_mlarray, cli_print_header, convert_image, header


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


def _medvol_2():
    try:
        import medvol
    except ImportError:
        return False
    return str(medvol.__version__).split(".")[0].isdigit() and int(str(medvol.__version__).split(".")[0]) >= 2


def _nifti(path, qform_shift=0.0):
    import nibabel as nib
    from nibabel.openers import ImageOpener
    from nibabel.volumeutils import array_to_file

    data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6, order="F")
    affine = np.array([[0.0, 0.0, 2.0, -10.0], [-0.7, 0.0, 0.0, 20.0], [0.0, 0.9, 0.0, -5.0], [0.0, 0.0, 0.0, 1.0]])
    header = nib.Nifti1Header()
    header.set_data_dtype(data.dtype)
    header.set_data_shape(data.shape)
    header.set_xyzt_units("mm")
    shifted = affine.copy()
    shifted[2, 3] += qform_shift
    header.set_qform(shifted, code=1)
    header.set_sform(affine, code=1)
    header["descrip"] = "cli fixture"
    with ImageOpener(str(path), "wb") as handle:
        header["vox_offset"] = 352
        header.write_to(handle)
        array_to_file(data, handle, data.dtype, offset=None, order="F")
    return data, affine


def _convert(argv):
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["mlarray_convert", *argv]), contextlib.redirect_stdout(out):
        cli_convert_to_mlarray()
    return out.getvalue()


@unittest.skipUnless(_medvol_2(), "needs MedVol 2")
class TestConvertWithMedVol(unittest.TestCase):
    def test_nifti_round_trip_keeps_data_geometry_and_header(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as tmpdir:
            source, mla, target = Path(tmpdir) / "a.nii.gz", Path(tmpdir) / "a.mla", Path(tmpdir) / "b.nii.gz"
            data, affine = _nifti(source)

            printed = _convert([str(source), str(mla)])
            self.assertIn("convert nifti -> mla", printed)
            image = MLArray(mla)
            self.assertEqual(image.shape, data.shape[::-1])
            self.assertTrue(np.array_equal(image.to_numpy(), data.transpose(2, 1, 0)))
            self.assertEqual(image.source[0].header["descrip"], "cli fixture")
            lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine
            np.testing.assert_allclose(image.matrix(), lps, atol=1e-6)   # array axes x, y, z = NIfTI i, j, k
            self.assertEqual(image.history[0]["action"], "converted")

            _convert([str(mla), str(target)])
            written = nib.load(str(target))
            self.assertTrue(np.array_equal(np.asarray(written.dataobj), data))
            np.testing.assert_allclose(written.header.get_sform(), affine, atol=1e-6)
            self.assertEqual(written.header["descrip"].item(), b"cli fixture")

    def test_policy_flags_are_passed_to_medvol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "a.nii.gz"
            _, affine = _nifti(source, qform_shift=5.0)
            printed = _convert([str(source), str(Path(tmpdir) / "q.mla"), "--nifti-form", "qform"])
            self.assertIn("nifti_form = qform", printed)
            self.assertIn("disagree", printed)
            self.assertAlmostEqual(MLArray(Path(tmpdir) / "q.mla").origin[2], affine[2, 3] + 5.0, places=5)


class TestConvertWithoutMedVol2(unittest.TestCase):
    def test_old_medvol_gives_a_clear_error(self):
        fake = mock.MagicMock(__version__="0.0.22")
        with mock.patch.dict(sys.modules, {"medvol": fake}):
            with self.assertRaisesRegex(RuntimeError, "MedVol 2"):
                convert_image("a.nii.gz", "b.mla")

    def test_missing_dependency_of_medvol_is_not_hidden(self):
        error = ImportError("No module named 'nrrd'", name="nrrd")
        with mock.patch.dict(sys.modules, {"medvol": None}), mock.patch("builtins.__import__", side_effect=error):
            with self.assertRaisesRegex(ImportError, "nrrd"):
                convert_image("a.nii.gz", "b.mla")

    def test_conversion_errors_exit_cleanly(self):
        with mock.patch("mlarray.cli.convert_image", side_effect=ValueError("No reader recognises x.foo")):
            with mock.patch.object(sys, "argv", ["mlarray_convert", "x.foo", "y.mla"]), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as exit_info:
                    cli_convert_to_mlarray()
        self.assertEqual(exit_info.exception.code, 1)
        self.assertIn("No reader recognises", err.getvalue())


if __name__ == "__main__":
    unittest.main()
