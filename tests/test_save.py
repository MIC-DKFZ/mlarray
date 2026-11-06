import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import blosc2
import numpy as np

from mlarray import MLArray


def _make_array(shape=(8, 16, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


class TestSave(unittest.TestCase):
    def test_save_opened_file_onto_itself_keeps_data_and_mode(self):
        for mode, mmap_mode in (("r", "r"), ("r", None), ("a", None), ("r", "r+")):
            with self.subTest(mode=mode, mmap_mode=mmap_mode), tempfile.TemporaryDirectory() as tmpdir:
                array = _make_array()
                path = Path(tmpdir) / "sample.mla"
                MLArray(array, spacing=(1, 2, 3), space="patient").save(path)

                opened = MLArray.open(path, mode=mode, mmap_mode=mmap_mode)
                opened.save(path)

                self.assertTrue(path.is_file())
                self.assertEqual(opened.mode, mode)
                self.assertEqual(opened.mmap_mode, mmap_mode)
                self.assertTrue(np.allclose(opened[...], array))
                reloaded = MLArray(path)
                self.assertTrue(np.allclose(reloaded.to_numpy(), array))
                self.assertEqual(reloaded.spacing, (1.0, 2.0, 3.0))
                self.assertEqual(sorted(os.listdir(tmpdir)), ["sample.mla"])

    def test_save_r_plus_onto_itself_persists_later_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=1)
            path = Path(tmpdir) / "sample.mla"
            MLArray(array, spacing=(1, 2, 3), space="patient").save(path)

            opened = MLArray.open(path, mmap_mode="r+")
            opened.set_geometry(spacing=[4, 5, 6])
            opened.save(path)
            opened[0, 0, 0] = 7
            opened.user["edited"] = True
            opened.close()

            reloaded = MLArray(path)
            self.assertEqual(reloaded[0, 0, 0], 7)
            self.assertEqual(reloaded.spacing, (4.0, 5.0, 6.0))
            self.assertEqual(reloaded.user["edited"], True)

    def test_save_created_file_onto_itself_does_not_truncate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=2)
            path = Path(tmpdir) / "created.mla"
            created = MLArray.create(path, shape=array.shape, dtype=array.dtype)
            created[...] = array
            created.save(path)

            self.assertEqual(created.mode, "a")
            self.assertEqual(created.mmap_mode, "r+")
            created[0, 0, 0] = 7
            created.close()

            reloaded = MLArray(path)
            expected = array.copy()
            expected[0, 0, 0] = 7
            self.assertTrue(np.allclose(reloaded.to_numpy(), expected))

    def test_save_opened_file_to_other_path_leaves_source_untouched(self):
        for mmap_mode in ("r", None, "r+"):
            with self.subTest(mmap_mode=mmap_mode), tempfile.TemporaryDirectory() as tmpdir:
                array = _make_array(seed=3)
                src = Path(tmpdir) / "src.mla"
                dst = Path(tmpdir) / "dst.mla"
                MLArray(array, spacing=(1, 2, 3), space="patient").save(src)

                opened = MLArray.open(src, mmap_mode=mmap_mode)
                # Read after opening: opening with r+ rewrites the metadata itself.
                src_bytes = src.read_bytes()
                if mmap_mode == "r+":
                    opened.set_geometry(spacing=[4, 5, 6])
                opened.save(dst)

                self.assertEqual(src.read_bytes(), src_bytes)
                saved = MLArray(dst)
                self.assertTrue(np.allclose(saved.to_numpy(), array))
                expected_spacing = (4.0, 5.0, 6.0) if mmap_mode == "r+" else (1.0, 2.0, 3.0)
                self.assertEqual(saved.spacing, expected_spacing)
                self.assertEqual(opened.mmap_mode, mmap_mode)

    def test_failed_save_keeps_existing_target_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=4)
            path = Path(tmpdir) / "sample.mla"
            MLArray(array).save(path)
            original_bytes = path.read_bytes()

            opened = MLArray.open(path)
            with mock.patch.object(MLArray, "_serialize_metadata", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    opened.save(path)

            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(sorted(os.listdir(tmpdir)), ["sample.mla"])

    def test_save_loaded_array_onto_its_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=5)
            path = Path(tmpdir) / "sample.mla"
            MLArray(array).save(path)

            loaded = MLArray(path)
            loaded[0, 0, 0] = 7
            loaded.save(path)

            reloaded = MLArray(path)
            self.assertEqual(reloaded[0, 0, 0], 7)
            self.assertIsNone(loaded.mode)
            self.assertIsNone(loaded.mmap_mode)

    def test_save_numpy_backed_keeps_instance_and_patch_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=7)
            path = Path(tmpdir) / "numpy.mla"
            image = MLArray(array, spacing=(1, 2, 3), space="patient", patch_size=(4, 8, 8), compressed=False)
            meta_before = image.meta.to_dict(include_lazy=True)

            image.save(path)

            self.assertEqual(image._backend, "numpy")
            self.assertIsInstance(image._store, np.ndarray)
            self.assertEqual(image.meta.to_dict(include_lazy=True), meta_before)
            reloaded = MLArray(path)
            self.assertTrue(np.array_equal(reloaded.to_numpy(), array))
            self.assertEqual(reloaded.meta.storage.patch_size, (4, 8, 8))
            self.assertEqual(reloaded.meta.storage.chunk_size, tuple(reloaded._store.chunks))
            self.assertEqual(reloaded.meta.storage.block_size, tuple(reloaded._store.blocks))
            self.assertEqual(reloaded.spacing, (1.0, 2.0, 3.0))

    def test_save_numpy_backed_matches_compressed_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array = _make_array(seed=8)
            MLArray(array, patch_size=(4, 8, 8)).save(Path(tmpdir) / "blosc2.mla")
            MLArray(array, patch_size=(4, 8, 8), compressed=False).save(Path(tmpdir) / "numpy.mla")

            compressed = MLArray(Path(tmpdir) / "blosc2.mla")
            uncompressed = MLArray(Path(tmpdir) / "numpy.mla")
            self.assertEqual(uncompressed.meta.storage.to_dict(), compressed.meta.storage.to_dict())
            self.assertEqual(uncompressed._store.schunk.cparams, compressed._store.schunk.cparams)

    def test_save_without_array_keeps_instance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image = MLArray(axes=["z", "y", "x"], spacing=(1, 2, 3), space="patient", compressed=False)
            store = image._store
            image.save(Path(tmpdir) / "meta_only.mla")
            self.assertIs(image._store, store)
            self.assertEqual(MLArray(Path(tmpdir) / "meta_only.mla").spacing, (1.0, 2.0, 3.0))

    def test_persisted_cparams_have_no_machine_specific_fields(self):
        cparams_object = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=3, nthreads=2)
        cparams_dict = {"codec": blosc2.Codec.ZSTD, "clevel": 3, "nthreads": 2, "typesize": 8, "blocksize": 4096}
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, cparams in (("object", cparams_object), ("dict", cparams_dict)):
                with self.subTest(cparams=name):
                    path = Path(tmpdir) / f"{name}.mla"
                    image = MLArray(_make_array(seed=9), cparams=cparams)
                    self.assertEqual(image._store.schunk.cparams.nthreads, 2)
                    image.save(path)

                    persisted = MLArray(path).meta.storage.to_dict()["compression"]
                    self.assertEqual(persisted, {"codec": "ZSTD", "level": 3} if name == "dict"
                                     else {"codec": "ZSTD", "level": 3, "filters": ["SHUFFLE"]})

    def test_create_compresses_with_given_cparams_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "created.mla"
            cparams = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=3, nthreads=2)
            created = MLArray.create(path, shape=(8, 16, 16), dtype=np.float32, cparams=cparams)
            self.assertEqual(created._store.schunk.cparams.nthreads, 2)
            self.assertEqual(created.meta.storage.to_dict()["compression"], {"codec": "ZSTD", "level": 3, "filters": ["SHUFFLE"]})
            created.close()
            self.assertEqual(MLArray.open(path).meta.storage.to_cparams().nthreads, 4)


if __name__ == "__main__":
    unittest.main()
