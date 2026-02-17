import tempfile
import unittest
from pathlib import Path

import blosc2
import numpy as np

from mlarray import MLArray


def _make_array(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestCreate(unittest.TestCase):
    def test_create_writable_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shape = (8, 16, 16)
            path = Path(tmpdir) / "created.mla"
            image = MLArray.create(
                path,
                shape=shape,
                dtype=np.float32,
                patch_size=None,
                chunk_size=(1, 8, 8),
                block_size=(1, 4, 4),
                mmap_mode="w+",
            )

            data = _make_array(shape=shape, seed=1)
            image[...] = data
            image.close()

            loaded = MLArray(path)
            self.assertEqual(loaded.shape, shape)
            self.assertEqual(loaded.dtype, np.float32)
            self.assertTrue(np.allclose(loaded[...], data))

    def test_create_persists_meta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "with-meta.mla"
            image = MLArray.create(
                path,
                shape=(4, 8, 8),
                dtype=np.float32,
                meta={"created_by": "test_create"},
                patch_size=None,
                chunk_size=(1, 4, 4),
                block_size=(1, 2, 2),
            )
            image.close()

            loaded = MLArray(path)
            self.assertEqual(loaded.meta.source["created_by"], "test_create")

    def test_create_accepts_cparams_dparams_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dict-params.mla"
            image = MLArray.create(
                path,
                shape=(4, 8, 8),
                dtype=np.float32,
                patch_size=None,
                chunk_size=(1, 4, 4),
                block_size=(1, 2, 2),
                cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 1},
                dparams={"nthreads": 1},
            )
            self.assertEqual(image.mode, "w")
            self.assertEqual(image.mmap_mode, "w+")
            image.close()

    def test_create_accepts_cparams_dparams_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object-params.mla"
            image = MLArray.create(
                path,
                shape=(4, 8, 8),
                dtype=np.float32,
                patch_size=None,
                chunk_size=(1, 4, 4),
                block_size=(1, 2, 2),
                cparams=blosc2.CParams(codec=blosc2.Codec.LZ4HC, clevel=1),
                dparams=blosc2.DParams(nthreads=1),
            )
            self.assertEqual(image.shape, (4, 8, 8))
            image.close()

    def test_create_raises_for_invalid_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid-mode.mla"
            with self.assertRaises(RuntimeError):
                MLArray.create(path, shape=(4, 8, 8), dtype=np.float32, mode="a")

    def test_create_raises_for_invalid_mmap_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid-mmap.mla"
            with self.assertRaises(RuntimeError):
                MLArray.create(path, shape=(4, 8, 8), dtype=np.float32, mmap_mode="r+")

    def test_create_raises_for_invalid_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid-extension.txt"
            with self.assertRaises(RuntimeError):
                MLArray.create(path, shape=(4, 8, 8), dtype=np.float32)


if __name__ == "__main__":
    unittest.main()
