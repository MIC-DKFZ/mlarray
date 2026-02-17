import tempfile
import unittest
from pathlib import Path

import blosc2
import numpy as np

from mlarray import MLArray


def _make_array(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestOpen(unittest.TestCase):
    def test_open_reads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array()
            path = Path(tmpdir) / "sample.mla"
            MLArray(array, meta={"case": "open-basic"}).save(path)

            opened = MLArray.open(path, mmap_mode="r")

            self.assertEqual(opened.mode, "r")
            self.assertEqual(opened.mmap_mode, "r")
            self.assertEqual(opened.shape, array.shape)
            self.assertTrue(np.allclose(opened[...], array))

    def test_open_accepts_dparams_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=1)
            path = Path(tmpdir) / "dict-dparams.mla"
            MLArray(array).save(path)

            opened = MLArray.open(path, dparams={"nthreads": 1}, mmap_mode="r")
            self.assertTrue(np.allclose(opened.to_numpy(), array))

    def test_open_accepts_dparams_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=2)
            path = Path(tmpdir) / "object-dparams.mla"
            MLArray(array).save(path)

            opened = MLArray.open(path, dparams=blosc2.DParams(nthreads=1), mmap_mode="r")
            self.assertTrue(np.allclose(opened.to_numpy(), array))

    def test_open_mode_a_with_mmap_none_persists_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=3)
            path = Path(tmpdir) / "append-mode.mla"
            MLArray(array, meta={"edited": False}).save(path)

            opened = MLArray.open(path, mode="a", mmap_mode=None)
            opened.meta.source["edited"] = True
            opened.close()

            reloaded = MLArray(path)
            self.assertEqual(reloaded.meta.source["edited"], True)

    def test_open_raises_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.mla"
            with self.assertRaises(RuntimeError):
                MLArray.open(path)

    def test_open_raises_for_invalid_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=4)
            path = Path(tmpdir) / "invalid-mode.mla"
            MLArray(array).save(path)

            with self.assertRaises(RuntimeError):
                MLArray.open(path, mode="w")

    def test_open_raises_for_invalid_mmap_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=5)
            path = Path(tmpdir) / "invalid-mmap.mla"
            MLArray(array).save(path)

            with self.assertRaises(RuntimeError):
                MLArray.open(path, mmap_mode="w+")

    def test_open_raises_for_invalid_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "not-mlarray.txt"
            with self.assertRaises(RuntimeError):
                MLArray.open(path)


if __name__ == "__main__":
    unittest.main()
