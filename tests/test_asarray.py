import tempfile
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

        mocked_blosc_asarray.assert_called_once()
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


if __name__ == "__main__":
    unittest.main()
