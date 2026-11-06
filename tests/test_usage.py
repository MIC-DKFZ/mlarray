import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlarray import MLArray, MLARRAY_DEFAULT_PATCH_SIZE
from mlarray.metadata import MetadataError


def _make_array(shape=(16, 32, 32), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestUsage(unittest.TestCase):
    def test_default_usage_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            image = MLArray(array)

            path = Path(tmpdir) / "sample.mla"
            image.save(path)

            loaded = MLArray(path)
            self.assertEqual(loaded.shape, array.shape)

    def test_default_usage_uncompressed_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=11)
            image = MLArray(array, compressed=False)
            self.assertTrue(isinstance(image._store, np.ndarray))

            path = Path(tmpdir) / "sample-uncompressed.mla"
            image.save(path)
            loaded = MLArray(path)
            self.assertEqual(loaded.shape, array.shape)
            self.assertTrue(np.allclose(loaded.to_numpy(), array))

    def test_mmap_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "sample.mla"
            MLArray(array).save(path)

            loaded = MLArray.open(path, mmap_mode="r")
            self.assertFalse(isinstance(loaded._store, np.ndarray))

    def test_load_uncompressed_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "sample-load-uncompressed.mla"
            MLArray(array).save(path)

            loaded = MLArray.load(path, compressed=False)
            self.assertTrue(isinstance(loaded._store, np.ndarray))
            self.assertTrue(np.allclose(loaded.to_numpy(), array))

    def test_loading_and_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            src = Path(tmpdir) / "sample.mla"
            dst = Path(tmpdir) / "copy.mla"

            MLArray(array).save(src)
            MLArray(src).save(dst)

            self.assertTrue(dst.exists())

    def test_metadata_inspection_and_manipulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            image = MLArray(
                array,
                axes=["z", "y", "x"],
                spacing=(1.5, 1.0, 1.0),
                origin=(10.0, 10.0, 30.0),
                direction=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                space="patient",
                values={"kind": "measurement", "unit": "hounsfield_unit"},
                user={"patient_id": "123", "modality": "CT"},
            )
            image.set_geometry(spacing=(1.5, 5.3, 1.0))
            image.user["study_id"] = "study-001"

            path = Path(tmpdir) / "with-metadata.mla"
            image.save(path)

            loaded = MLArray(path)
            self.assertEqual(loaded.spacing, (1.5, 5.3, 1.0))
            self.assertEqual(loaded.origin, (10.0, 10.0, 30.0))
            self.assertEqual(loaded.letter_codes(), ("S", "P", "L"))
            self.assertEqual(loaded.values.unit_code, "[hnsf'U]")
            self.assertEqual(loaded.user["study_id"], "study-001")

    def test_geometry_is_derived_and_changed_through_set_geometry(self):
        image = MLArray(_make_array(shape=(8, 8, 8)))
        self.assertIsNone(image.spacing)
        with self.assertRaises(AttributeError):
            image.spacing = [0.1, 0.2, 0.3]

        image.set_geometry(spacing=[0.1, 0.2, 0.3], origin=[1.0, 2.0, 3.0], space="patient")
        self.assertEqual(image.spacing, (0.1, 0.2, 0.3))
        self.assertEqual(image.origin, (1.0, 2.0, 3.0))
        np.testing.assert_array_equal(image.direction, np.eye(3))
        image.set_geometry(origin=[0.0, 0.0, 0.0])
        self.assertEqual(image.spacing, (0.1, 0.2, 0.3))
        np.testing.assert_allclose(image.matrix(), np.diag([0.1, 0.2, 0.3, 1.0]))

    def test_explicit_space_with_matrix(self):
        matrix = [
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [-3.0, 0.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        image = MLArray(
            _make_array(shape=(8, 8, 8)),
            axes=["z", "y", "x"],
            spaces=[{
                "name": "patient",
                "axes": [
                    {"name": "x", "unit": "millimeter", "increases_toward": "left"},
                    {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
                    {"name": "z", "unit": "millimeter", "increases_toward": "superior"},
                ],
                "from_array": {"array_axes": ["z", "y", "x"], "matrix": matrix},
            }],
        )
        np.testing.assert_array_equal(image.matrix(), matrix)
        self.assertEqual(image.spacing, (3.0, 2.0, 1.0))
        self.assertEqual(image.origin, (10.0, 20.0, 30.0))
        self.assertEqual(image.letter_codes(), ("I", "P", "L"))

    def test_geometry_needs_a_declared_space(self):
        array = _make_array(shape=(8, 8, 8))
        with self.assertRaises(MetadataError):
            MLArray(array, spacing=(1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            MLArray(array, spacing=(1.0, 1.0, 1.0), space="patient", spaces=[])
        with self.assertRaises(MetadataError):
            MLArray(array).matrix()

    def test_copy_metadata_with_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            base = MLArray(
                array,
                spacing=(1.0, 1.0, 1.0),
                origin=(1.0, 2.0, 3.0),
                space="patient",
                values={"kind": "label", "labels": {"1": {"name": "liver"}}},
                user={"case": "base"},
                source=[{"format": "nifti", "header": {"descrip": "base"}}],
                profiles={"ml": {"statistics": {"min": 0, "max": 1}, "split": "train"}},
            )
            base_path = Path(tmpdir) / "base.mla"
            base.save(base_path)

            base_loaded = MLArray(base_path)
            array2 = _make_array(seed=1)
            image = MLArray(array2, spacing=(0.8, 0.8, 1.0), space="patient", copy=base_loaded)

            self.assertEqual(image.spacing, (0.8, 0.8, 1.0))
            self.assertEqual(image.origin, base_loaded.origin)
            np.testing.assert_array_equal(image.direction, base_loaded.direction)
            self.assertEqual(image.values, base_loaded.values)
            self.assertEqual(image.user, {"case": "base"})
            self.assertEqual(image.profiles["ml"], {"split": "train"})
            self.assertEqual(image.source, ())
            with_source = MLArray(array2, copy=base_loaded, copy_source=True)
            self.assertEqual(with_source.source[0].header["descrip"], "base")

    def test_patch_size_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "default-patch.mla"
            MLArray(array).save(path)

            loaded = MLArray(path)
            self.assertEqual(
                loaded.meta.storage.patch_size,
                (MLARRAY_DEFAULT_PATCH_SIZE,) * 3,
            )

    def test_patch_size_isotropic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "patch-64.mla"
            MLArray(array, patch_size=64).save(path)

            loaded = MLArray(path)
            self.assertEqual(loaded.meta.storage.patch_size, (64, 64, 64))

    def test_patch_size_non_isotropic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "patch-non-iso.mla"
            MLArray(array, patch_size=[16, 24, 32]).save(path)

            loaded = MLArray(path)
            self.assertEqual(loaded.meta.storage.patch_size, (16, 24, 32))

    def test_manual_chunk_and_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "manual-chunk-block.mla"
            MLArray(
                array,
                patch_size=None,
                chunk_size=(1, 16, 16),
                block_size=(1, 8, 8),
            ).save(path)

            loaded = MLArray(path)
            self.assertEqual(loaded.meta.storage.chunk_size, (1, 16, 16))
            self.assertEqual(loaded.meta.storage.block_size, (1, 8, 8))

    def test_b2nd_extension_not_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "plain.b2nd"
            image = MLArray(array)

            with self.assertRaises(RuntimeError):
                image.save(path)
            with self.assertRaises(RuntimeError):
                MLArray(path)


if __name__ == "__main__":
    unittest.main()
