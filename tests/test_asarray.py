import tempfile
import warnings
import unittest
from pathlib import Path
from unittest.mock import patch

import blosc2
import numpy as np

from mlarray import MLArray
from mlarray.meta import Meta


def _make_array(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestAsArray(unittest.TestCase):
    def test_asarray_creates_blosc2_store_and_meta(self):
        array = _make_array()
        image = MLArray.asarray(array, meta={"case_id": "plain"})

        self.assertFalse(isinstance(image._store, np.ndarray))
        self.assertEqual(image.shape, array.shape)
        self.assertTrue(np.allclose(image.to_numpy(), array))
        self.assertEqual(image.meta.source.to_plain(), {"case_id": "plain"})

    def test_asarray_calls_blosc2_asarray(self):
        array = _make_array()
        with patch("mlarray.mlarray.blosc2.asarray", wraps=blosc2.asarray) as mocked_blosc_asarray:
            image = MLArray.asarray(array, dparams={"nthreads": 2})

        # the last call builds the store; earlier ones are the filter choice's trial compressions
        self.assertEqual(mocked_blosc_asarray.call_args.kwargs["dparams"], {"nthreads": 2})
        self.assertFalse(isinstance(image._store, np.ndarray))
        self.assertTrue(np.allclose(image.to_numpy(), array))

    def test_asarray_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=1)
            image = MLArray.asarray(
                array,
                patch_size=None,
                chunk_size=(1, 8, 8),
                block_size=(1, 4, 4),
                cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 3},
                dparams={"nthreads": 1},
            )

            self.assertFalse(isinstance(image._store, np.ndarray))
            self.assertTrue(np.allclose(image.to_numpy(), array))
            self.assertEqual(image.meta.blosc2.chunk_size, [1, 8, 8])
            self.assertEqual(image.meta.blosc2.block_size, [1, 4, 4])

            path = Path(tmpdir) / "asarray-compressed.mla"
            image.save(path)
            loaded = MLArray(path)
            self.assertTrue(np.allclose(loaded.to_numpy(), array))

    def test_asarray_accepts_meta_object(self):
        array = _make_array(seed=2)
        meta = Meta(source={"patient_id": "p-001"}, is_seg=True)
        image = MLArray.asarray(array, meta=meta)

        self.assertEqual(image.meta.source.to_plain(), {"patient_id": "p-001"})
        self.assertTrue(image.meta.is_seg)

    def test_asarray_uncompressed_numpy_store_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=3)
            image = MLArray.asarray(array, meta={"case_id": "np"}, compressed=False)

            self.assertTrue(isinstance(image._store, np.ndarray))
            self.assertTrue(np.allclose(image.to_numpy(), array))
            self.assertEqual(image.meta.source.to_plain(), {"case_id": "np"})

            path = Path(tmpdir) / "asarray-numpy-save.mla"
            image.save(path)
            loaded = MLArray(path)
            self.assertFalse(isinstance(loaded._store, np.ndarray))
            self.assertTrue(np.allclose(loaded.to_numpy(), array))
            self.assertEqual(loaded.meta.source.to_plain(), {"case_id": "np"})


    def test_asarray_keeps_zero_dimensional_shape(self):
        image = MLArray.asarray(np.array(5.0), patch_size=None)
        self.assertEqual(image.to_numpy().shape, ())
        self.assertEqual(float(image[()]), 5.0)

    def test_numpy_conversion_honors_copy(self):
        array = _make_array(seed=7)
        compressed = MLArray.asarray(array)
        uncompressed = MLArray.asarray(array, compressed=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertTrue(np.allclose(np.asarray(compressed), array))
            self.assertEqual(np.asarray(compressed, dtype=np.float64).dtype, np.float64)
        with self.assertRaises(ValueError):
            np.asarray(compressed, copy=False)

        self.assertTrue(np.shares_memory(np.asarray(uncompressed, copy=False), uncompressed._store))
        self.assertFalse(np.shares_memory(np.array(uncompressed, copy=True), uncompressed._store))

    def test_empty_arrays_report_their_shape(self):
        cases = {
            "asarray": lambda: MLArray.asarray(np.zeros((0,), dtype=np.float32)),
            "linspace": lambda: MLArray.linspace(0, 1, 0),
            "arange": lambda: MLArray.arange(5, 0),
            "uncompressed": lambda: MLArray(np.zeros((0,), dtype=np.float32), compressed=False),
        }
        for name, make in cases.items():
            with self.subTest(case=name):
                image = make()
                self.assertEqual(image.shape, (0,))
                self.assertEqual(image.ndim, 1)
                self.assertEqual(image.to_numpy().shape, (0,))

    def test_zero_length_axes_save_and_load(self):
        cases = {
            "compressed": lambda: MLArray(np.zeros((0, 10, 10), dtype=np.float32), spacing=(1, 2, 3)),
            "uncompressed": lambda: MLArray(np.zeros((0, 10, 10), dtype=np.float32), spacing=(1, 2, 3), compressed=False),
        }
        for name, make in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "empty.mla"
                image = make()
                self.assertIsNone(image.meta.blosc2.patch_size)
                image.save(path)
                loaded = MLArray(path)
                self.assertEqual(loaded.shape, (0, 10, 10))
                self.assertEqual(loaded.to_numpy().shape, (0, 10, 10))
                self.assertEqual(loaded.spacing, [1.0, 2.0, 3.0])

    def test_create_with_zero_length_axis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "created.mla"
            image = MLArray.create(path, shape=(4, 0, 3), dtype=np.uint8)
            image.close()
            self.assertEqual(MLArray.open(path).shape, (4, 0, 3))


if __name__ == "__main__":
    unittest.main()
