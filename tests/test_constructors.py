import tempfile
import unittest
from pathlib import Path

import blosc2
import numpy as np

from mlarray import MLArray


class TestInMemoryConstructors(unittest.TestCase):
    def test_empty_and_save_roundtrip(self):
        shape = (4, 8, 8)
        arr = MLArray.empty(
            shape=shape,
            dtype=np.float32,
            patch_size=None,
            chunk_size=(1, 4, 4),
            block_size=(1, 2, 2),
            num_threads=1,
            cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 1},
            dparams={"nthreads": 1},
        )

        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        arr[...] = data

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "in-memory-empty.mla"
            arr.save(path)
            loaded = MLArray(path)

            self.assertEqual(loaded.shape, shape)
            self.assertTrue(np.allclose(loaded[...], data))

    def test_zeros_ones_full(self):
        shape = (3, 5)

        zeros = MLArray.zeros(
            shape=shape,
            dtype=np.int32,
            patch_size=None,
            chunk_size=(3, 5),
            block_size=(1, 5),
        )
        ones = MLArray.ones(
            shape=shape,
            dtype=np.float32,
            patch_size=None,
            chunk_size=(3, 5),
            block_size=(1, 5),
        )
        full = MLArray.full(
            shape=shape,
            fill_value=7,
            dtype=np.int16,
            patch_size=None,
            chunk_size=(3, 5),
            block_size=(1, 5),
        )

        self.assertTrue(np.array_equal(zeros[...], np.zeros(shape, dtype=np.int32)))
        self.assertTrue(np.array_equal(ones[...], np.ones(shape, dtype=np.float32)))
        self.assertTrue(np.array_equal(full[...], np.full(shape, 7, dtype=np.int16)))

    def test_arange_and_linspace(self):
        arr1 = MLArray.arange(0, 12, 2, patch_size=None, num_threads=1)
        self.assertTrue(np.array_equal(arr1[...], np.arange(0, 12, 2)))

        arr2 = MLArray.arange(0, 12, 2, shape=(2, 3), patch_size=None, num_threads=1)
        self.assertTrue(np.array_equal(arr2[...], np.arange(0, 12, 2).reshape(2, 3)))

        lin = MLArray.linspace(
            0.0, 1.0, num=6, shape=(2, 3), endpoint=False, patch_size=None, num_threads=1
        )
        self.assertTrue(
            np.allclose(lin[...], np.linspace(0.0, 1.0, 6, endpoint=False).reshape(2, 3))
        )

    def test_like_methods_and_meta_copy(self):
        base = MLArray.full(
            shape=(2, 3, 4),
            fill_value=5,
            dtype=np.int16,
            meta={"case": "meta-copy"},
            patch_size=None,
            chunk_size=(1, 3, 4),
            block_size=(1, 2, 2),
        )

        empty_like = MLArray.empty_like(
            base, patch_size=None, chunk_size=(1, 3, 4), block_size=(1, 2, 2)
        )
        zeros_like = MLArray.zeros_like(
            base, patch_size=None, chunk_size=(1, 3, 4), block_size=(1, 2, 2)
        )
        ones_like = MLArray.ones_like(
            base,
            dtype=np.float32,
            patch_size=None,
            chunk_size=(1, 3, 4),
            block_size=(1, 2, 2),
        )
        full_like = MLArray.full_like(
            base,
            fill_value=9,
            patch_size=None,
            chunk_size=(1, 3, 4),
            block_size=(1, 2, 2),
        )

        self.assertEqual(empty_like.shape, base.shape)
        self.assertEqual(zeros_like.shape, base.shape)
        self.assertEqual(ones_like.shape, base.shape)
        self.assertEqual(full_like.shape, base.shape)

        self.assertTrue(
            np.array_equal(zeros_like[...], np.zeros(base.shape, dtype=base.dtype))
        )
        self.assertTrue(
            np.array_equal(ones_like[...], np.ones(base.shape, dtype=np.float32))
        )
        self.assertTrue(
            np.array_equal(full_like[...], np.full(base.shape, 9, dtype=base.dtype))
        )

        self.assertEqual(zeros_like.meta.source.to_plain(), {"case": "meta-copy"})

    def test_like_raises_for_invalid_input(self):
        with self.assertRaises(TypeError):
            MLArray.zeros_like(123)


if __name__ == "__main__":
    unittest.main()
