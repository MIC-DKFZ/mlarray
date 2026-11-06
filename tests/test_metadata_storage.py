import copy
import gzip
import pickle
import tempfile
import time
import unittest
from pathlib import Path

import blosc2
import msgpack
import numpy as np

from mlarray import MLArray
from mlarray import metadata_io
from test_schemas import validate_core, validate_source_key, validate_user_key

SOURCE = [{"format": "nifti", "files": ["a.nii.gz"], "header": {"descrip": "scan", "pixdim": [1.0, 0.7, 0.7, 2.5], "big": list(range(2000))}}]


class _CreatesMarkerFile:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _make_array(shape=(4, 8, 8)):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def _make_file(path, **kwargs):
    kwargs.setdefault("spacing", (1, 2, 3))
    kwargs.setdefault("space", "patient")
    MLArray(_make_array(), **kwargs).save(path)


def _raw(path, key):
    store = blosc2.open(str(path))  # keep the array alive while its vlmeta is used
    return msgpack.unpackb(store.schunk.vlmeta.get_vlmeta(key), raw=False, strict_map_key=False)


def _keys(path):
    store = blosc2.open(str(path))
    return set(store.schunk.vlmeta.keys())


class TestFileLayout(unittest.TestCase):
    def test_three_keys_of_plain_msgpack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.mla"
            _make_file(path, user={"case": ("a", 1)}, source=SOURCE)

            core = _raw(path, "mlarray")
            self.assertEqual(core["version"], "1.0")
            self.assertTrue(core["writer"].startswith("mlarray "))
            self.assertEqual([axis["name"] for axis in core["axes"]], ["axis_0", "axis_1", "axis_2"])
            self.assertEqual(core["spaces"][0]["axes"][0]["increases_toward"], "left")
            self.assertNotIn("source", core)
            self.assertNotIn("user", core)
            self.assertEqual(_raw(path, "mlarray.user"), {"user": {"case": ["a", 1]}})
            self.assertEqual(_raw(path, "mlarray.source")["source"][0]["header"]["descrip"], "scan")

    def test_written_files_validate_against_the_schemas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.mla"
            _make_file(
                path,
                axes=["z", "y", "x"],
                values={"kind": "label", "labels": {"1": {"name": "liver"}}},
                profiles={"ml": {"profile_version": "1.0", "split": "train"}},
                user={"case": 1},
                source=SOURCE,
            )
            validate_core(_raw(path, "mlarray"))
            validate_source_key(_raw(path, "mlarray.source"))
            validate_user_key(_raw(path, "mlarray.user"))

    def test_no_keys_for_empty_source_and_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.mla"
            _make_file(path)
            self.assertEqual(_keys(path), {"mlarray"})


class TestSafeReading(unittest.TestCase):
    def test_pickled_metadata_is_not_executed(self):
        for opener in (MLArray, MLArray.open):
            with self.subTest(opener=opener.__name__), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "evil.mla"
                marker = Path(tmpdir) / "marker"
                _make_file(path)
                store = blosc2.open(str(path), mode="a")
                store.schunk.vlmeta["mlarray"] = gzip.compress(pickle.dumps(_CreatesMarkerFile(str(marker))))
                del store

                with self.assertRaises(metadata_io.MetadataFormatError):
                    opener(path)
                self.assertFalse(marker.exists())

    def test_msgpack_extension_types_are_refused(self):
        for key in ("mlarray", "mlarray.user"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "ext.mla"
                _make_file(path, user={"a": 1})
                store = blosc2.open(str(path), mode="a")
                store.schunk.vlmeta.set_vlmeta(key, msgpack.packb({"user": msgpack.ExtType(42, b"payload")}), typesize=1)
                del store

                with self.assertRaises(metadata_io.MetadataFormatError):
                    MLArray(path).user

    def test_files_without_v1_metadata_are_refused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plain.mla"
            blosc2.asarray(_make_array(), urlpath=str(path), mode="w")
            with self.assertRaisesRegex(metadata_io.MetadataFormatError, "missing"):
                MLArray(path)


class TestLazyParts(unittest.TestCase):
    def _make(self, tmpdir):
        path = Path(tmpdir) / "sample.mla"
        _make_file(path, user={"note": "x"}, source=SOURCE)
        return path

    def test_open_defers_decoding_source_and_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make(tmpdir)
            for image in (MLArray.open(path), MLArray(path)):
                self.assertFalse(image.meta._source.loaded)
                self.assertFalse(image.meta._user.loaded)
                self.assertEqual(image.spacing, (1.0, 2.0, 3.0))
                self.assertEqual(image.source[0].header["big"], tuple(range(2000)))
                self.assertEqual(image.user, {"note": "x"})
                self.assertTrue(image.meta._source.loaded)

    def test_undecoded_parts_survive_rewrites_and_conversions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make(tmpdir)
            out = Path(tmpdir) / "out.mla"

            opened = MLArray.open(path, mmap_mode="r+")
            opened.set_geometry(spacing=[4, 5, 6])
            opened.save(out)
            opened.close()
            conversions = (
                lambda: MLArray(path),
                lambda: MLArray.load(path, compressed=False),
                lambda: _compressed(MLArray.open(path)),
                lambda: _decompressed(MLArray.open(path)),
                lambda: MLArray(_make_array(), copy=MLArray.open(path), copy_source=True),
            )
            for i, convert in enumerate(conversions):
                with self.subTest(conversion=i):
                    convert().save(out)
                    reloaded = MLArray(out)
                    self.assertEqual(reloaded.source[0].header["descrip"], "scan")
                    self.assertEqual(reloaded.user, {"note": "x"})
            self.assertEqual(MLArray(path).spacing, (4.0, 5.0, 6.0))

    def test_edited_user_is_written_and_empty_user_removes_the_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make(tmpdir)
            opened = MLArray.open(path, mmap_mode="r+")
            opened.user["added"] = 1
            opened.close()
            self.assertEqual(MLArray(path).user, {"note": "x", "added": 1})

            opened = MLArray.open(path, mmap_mode="r+")
            opened.user = {}
            opened.close()
            self.assertNotIn("mlarray.user", _keys(path))
            self.assertEqual(MLArray(path).user, {})

    def test_lazy_meta_can_be_pickled_and_copied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make(tmpdir)
            for clone in (pickle.loads(pickle.dumps(MLArray.open(path).meta)), copy.deepcopy(MLArray.open(path).meta)):
                self.assertEqual(clone.source[0].header["descrip"], "scan")
                self.assertEqual(clone.user, {"note": "x"})

    def test_opening_does_not_scale_with_source_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.mla"
            header = {f"field_{i}": [float(i)] * 50 for i in range(20000)}
            _make_file(path, source=[{"format": "dicom", "header": header}])
            start = time.perf_counter()
            for _ in range(10):
                MLArray.open(path)
            open_seconds = (time.perf_counter() - start) / 10
            start = time.perf_counter()
            MLArray.open(path).source
            decode_seconds = time.perf_counter() - start
            self.assertLess(open_seconds, decode_seconds)


class TestWriteModes(unittest.TestCase):
    def _saved(self, tmpdir):
        path = Path(tmpdir) / "sample.mla"
        _make_file(path, user={"a": 1})
        return path

    def test_read_only_and_copy_on_write_maps_do_not_write(self):
        for mmap_mode in ("r", "c"):
            with self.subTest(mmap_mode=mmap_mode), tempfile.TemporaryDirectory() as tmpdir:
                path = self._saved(tmpdir)
                opened = MLArray.open(path, mmap_mode=mmap_mode)
                opened.user["a"] = 2
                opened.close()
                self.assertEqual(MLArray(path).user["a"], 1)

    def test_read_write_map_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._saved(tmpdir)
            opened = MLArray.open(path, mmap_mode="r+")
            opened.user["a"] = 2
            opened.close()
            self.assertEqual(MLArray(path).user["a"], 2)

    def test_created_file_persists_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "create.mla"
            created = MLArray.create(path, shape=(8, 16, 16), dtype=np.float32, meta={"axes": ["z", "y", "x"]})
            created.user["created"] = True
            created.close()
            reloaded = MLArray(path)
            self.assertEqual(reloaded.user["created"], True)
            self.assertEqual([axis.name for axis in reloaded.axes], ["z", "y", "x"])

    def test_invalid_metadata_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._saved(tmpdir)
            opened = MLArray.open(path, mmap_mode="r+")
            opened.profiles["ml"] = {"boxes": {"frame": "index", "axes": ["nope"], "items": []}}
            with self.assertRaises(ValueError):
                opened.flush()
            self.assertNotIn("profiles", _raw(path, "mlarray"))


def _compressed(image):
    image.compress()
    return image


def _decompressed(image):
    image.decompress()
    return image


if __name__ == "__main__":
    unittest.main()
