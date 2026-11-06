import multiprocessing
import pickle
import tempfile
import unittest
from pathlib import Path

import blosc2
import numpy as np

from mlarray import MLArray


def _make_array(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def _read_patch(args):
    """DataLoader-style worker: receives a pickled MLArray and reads a patch."""
    image, index = args
    return index, np.asarray(image[index : index + 2, 2:10, 3:11]), image.spacing


class TestPickle(unittest.TestCase):
    def _assert_same(self, restored, original, array):
        self.assertTrue(np.array_equal(restored[...], array))
        self.assertEqual(restored.spacing, original.spacing)
        self.assertEqual(restored.meta.to_dict(include_lazy=True), original.meta.to_dict(include_lazy=True))

    def test_in_memory_blosc2_round_trip_keeps_layout_and_cparams(self):
        array = _make_array()
        image = MLArray(array, spacing=(1, 2, 3), space="patient", patch_size=(4, 8, 8))
        restored = pickle.loads(pickle.dumps(image))

        self._assert_same(restored, image, array)
        self.assertEqual(restored._backend, "blosc2")
        self.assertEqual(restored._store.chunks, image._store.chunks)
        self.assertEqual(restored._store.blocks, image._store.blocks)
        self.assertEqual(restored._store.schunk.cparams, image._store.schunk.cparams)
        self.assertEqual(restored._store.schunk.dparams, image._store.schunk.dparams)
        self.assertIsNone(restored.filepath)

    def test_numpy_backed_round_trip(self):
        array = _make_array(seed=1)
        image = MLArray(array, spacing=(1, 2, 3), space="patient", compressed=False)
        restored = pickle.loads(pickle.dumps(image))

        self._assert_same(restored, image, array)
        self.assertEqual(restored._backend, "numpy")

    def test_empty_instance_round_trip(self):
        restored = pickle.loads(pickle.dumps(MLArray()))
        self.assertIsNone(restored.shape)

    def test_opened_file_is_reopened_with_same_mode(self):
        for mode, mmap_mode in (("r", "r"), ("r", None), ("a", None), ("r", "r+")):
            with self.subTest(mode=mode, mmap_mode=mmap_mode), tempfile.TemporaryDirectory() as tmpdir:
                array = _make_array(seed=2)
                path = Path(tmpdir) / "sample.mla"
                MLArray(array, spacing=(1, 2, 3), space="patient").save(path)

                opened = MLArray.open(path, mode=mode, mmap_mode=mmap_mode, dparams={"nthreads": 2})
                restored = pickle.loads(pickle.dumps(opened))

                self._assert_same(restored, opened, array)
                self.assertEqual(restored.filepath, str(path))
                self.assertEqual((restored.mode, restored.mmap_mode), (mode, mmap_mode))
                self.assertEqual(restored._store.schunk.dparams.nthreads, 2)
                self.assertEqual(restored._store.urlpath, str(path))

    def test_created_file_is_reopened_without_truncating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=3)
            path = Path(tmpdir) / "created.mla"
            created = MLArray.create(path, shape=array.shape, dtype=array.dtype)
            created[...] = array

            restored = pickle.loads(pickle.dumps(created))

            self.assertTrue(np.array_equal(restored[...], array))
            self.assertEqual((restored.mode, restored.mmap_mode), ("a", "r+"))
            restored[0] = 0
            self.assertTrue(np.all(MLArray.open(path)[0] == 0))

    def test_unflushed_metadata_is_carried_over(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.mla"
            MLArray(_make_array(seed=4)).save(path)
            opened = MLArray.open(path, mmap_mode="r+")
            opened.set_geometry(spacing=[3, 3, 3], space="patient")

            restored = pickle.loads(pickle.dumps(opened))

            self.assertEqual(restored.spacing, (3.0, 3.0, 3.0))

    def test_spawn_workers_read_patches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=5)
            path = Path(tmpdir) / "sample.mla"
            MLArray(array, spacing=(1, 2, 3), space="patient").save(path)
            images = {
                "opened": MLArray.open(path),
                "in_memory": MLArray(array, spacing=(1, 2, 3), space="patient"),
                "numpy": MLArray(array, spacing=(1, 2, 3), space="patient", compressed=False),
            }
            context = multiprocessing.get_context("spawn")
            with context.Pool(2) as pool:
                for name, image in images.items():
                    with self.subTest(backing=name):
                        results = pool.map(_read_patch, [(image, i) for i in range(4)])
                        for index, patch, spacing in results:
                            self.assertTrue(np.array_equal(patch, array[index : index + 2, 2:10, 3:11]))
                            self.assertEqual(spacing, (1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
