"""The examples of spec sections 4-8 round-trip through save and load."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlarray import MLArray, fingerprint
from test_metadata_model import LABELS, ML, PATIENT

SHAPE = (10, 20, 30)


def _spec_image(array=None):
    array = np.arange(np.prod(SHAPE), dtype=np.int16).reshape(SHAPE) if array is None else array
    image = MLArray(
        array,
        axes=["z", "y", "x"],
        spaces=[PATIENT],
        values=LABELS,
        profiles={
            "medical": {"profile_version": "1.0", "dicom": {"Modality": "MR", "BodyPartExamined": "BRAIN", "RepetitionTime": 2300, "EchoTime": 2.98}},
            "ml": ML,
            "org_example_cryoem": {"grid": [1, 2]},
        },
        user={"anything": {"nested": [1, 2.5, None, True, "x", b"\x00"]}},
        source=[{
            "format": "nifti",
            "format_version": "1",
            "files": ["sub-01_T1w.nii.gz"],
            "reader": {"name": "medvol", "version": "2.0.0"},
            "index_map": {"axis_order": [2, 1, 0], "flipped": [False, False, False], "offset": [0, 0, 0]},
            "fingerprint": {"shape": [30, 20, 10], "dtype": "int16", "geometry": "xxh3_64:00ff", "data": "xxh3_128:00ff"},
            "header": {"sizeof_hdr": 348, "descrip": "T1w", "pixdim": [1.0, 0.7, 0.7, 2.5], "extensions": [{"code": 4, "content": b"\x01\x02"}]},
        }],
    )
    image.meta.add_history("converted", tool="medvol 2.0.0", time="2026-11-06T10:12:00Z", settings={"nifti_form": "nibabel"},
                           warnings=["qform and sform disagree by 4.9 mm; used sform (sform_code=2)"])
    return image


class TestSpecExamplesRoundTrip(unittest.TestCase):
    def test_spec_example_round_trips_through_every_reader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.mla"
            image = _spec_image()
            image.save(path)
            expected = image.meta.to_dict(include_lazy=True)
            for name, reader in (("load", MLArray), ("open", MLArray.open), ("numpy", lambda p: MLArray.load(p, compressed=False))):
                with self.subTest(reader=name):
                    loaded = reader(path)
                    self.assertEqual(loaded.meta.to_dict(include_lazy=True), expected)
                    self.assertTrue(np.array_equal(loaded.to_numpy(), image.to_numpy()))
                    np.testing.assert_array_equal(loaded.matrix(), np.array(PATIENT["from_array"]["matrix"]))
                    self.assertEqual(loaded.letter_codes(), ("I", "P", "L"))
                    self.assertEqual(loaded.source[0].header["extensions"][0]["content"], b"\x01\x02")
                    self.assertEqual(loaded.history[0]["warnings"][0][:20], "qform and sform disa")

    def test_resaving_keeps_everything(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first, second = Path(tmpdir) / "a.mla", Path(tmpdir) / "b.mla"
            _spec_image().save(first)
            MLArray(first).save(second)
            opened = MLArray.open(first, mmap_mode="r+")
            opened.close()
            self.assertEqual(MLArray(second).meta, MLArray(first).meta)

    def test_pickled_open_file_keeps_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.mla"
            _spec_image().save(path)
            opened = MLArray.open(path)
            restored = pickle.loads(pickle.dumps(opened))
            self.assertEqual(restored.meta, opened.meta)


class TestDerivedImages(unittest.TestCase):
    def test_like_constructors_copy_geometry_but_not_data_dependent_metadata(self):  # M13
        image = _spec_image()
        for name in ("zeros_like", "ones_like", "empty_like"):
            with self.subTest(constructor=name):
                derived = getattr(MLArray, name)(image)
                self.assertEqual(derived.spaces, image.spaces)
                self.assertEqual(derived.axes, image.axes)
                self.assertEqual(derived.values.kind, "measurement")
                self.assertIsNone(derived.values.labels)
                self.assertNotIn("statistics", derived.profiles["ml"])
                self.assertNotIn("boxes", derived.profiles["ml"])
                self.assertEqual(derived.profiles["medical"], image.profiles["medical"])
                self.assertEqual(derived.source, ())
                self.assertEqual(derived.history, ())
        self.assertIn("statistics", image.profiles["ml"])

    def test_copy_keeps_labels_and_needs_opt_in_for_source(self):
        image = _spec_image()
        derived = MLArray(np.zeros(SHAPE, dtype=np.uint8), copy=image)
        self.assertEqual(derived.values, image.values)
        self.assertEqual(derived.source, ())
        self.assertEqual(MLArray(np.zeros(SHAPE, dtype=np.uint8), copy=image, copy_source=True).source, image.source)


class TestFingerprint(unittest.TestCase):
    def test_fingerprint_survives_save_and_detects_edits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.mla"
            image = _spec_image()
            fp = image.fingerprint()
            image.save(path)

            loaded = MLArray.open(path, mmap_mode="r+")
            self.assertTrue(fingerprint.matches(fp, loaded, loaded.meta))
            loaded[0, 0, 0] = 1
            self.assertFalse(fingerprint.matches(fp, loaded, loaded.meta))


if __name__ == "__main__":
    unittest.main()
