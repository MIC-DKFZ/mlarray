import copy
import pickle
import unittest

import numpy as np

from mlarray import fingerprint
from mlarray.metadata import (
    Axis,
    Meta,
    MetadataError,
    SourceEntry,
    Space,
    Storage,
    Values,
    space_from_geometry,
)

SHAPE = (10, 20, 30)

PATIENT = {
    "name": "patient",
    "reference": {"scheme": "dicom_frame_of_reference", "id": "1.2.826.0.1.3680043.8.498.1"},
    "axes": [
        {"name": "x", "unit": "millimeter", "increases_toward": "left"},
        {"name": "y", "unit": "millimeter", "increases_toward": "posterior"},
        {"name": "z", "unit": "millimeter", "increases_toward": "superior"},
    ],
    "from_array": {
        "array_axes": ["z", "y", "x"],
        "matrix": [[0.0, 0.0, 0.7, -170.1], [0.0, 0.7, 0.0, -201.3], [-2.5, 0.0, 0.0, -87.0], [0.0, 0.0, 0.0, 1.0]],
    },
}

LABELS = {
    "kind": "label",
    "labels": {
        "0": {"name": "background"},
        "1": {"name": "liver", "color": [221, 130, 101], "code": {"scheme": "SCT", "value": "10200004", "meaning": "Liver"}},
    },
}

ML = {
    "profile_version": "1.0",
    "statistics": {"min": -1024, "max": 3071, "mean": 40.2, "standard_deviation": 410.0, "percentiles": {"0.5": -1000, "99.5": 1500}},
    "boxes": {"frame": "index", "axes": ["z", "y", "x"], "items": [{"min": [1, 2, 3], "max": [4, 5, 6], "label": 1, "score": 0.97}]},
    "patch_size": [8, 16, 16],
    "split": "train",
}


def _meta(**overrides):
    kwargs = dict(
        axes=["z", "y", "x"],
        spaces=[PATIENT],
        values=LABELS,
        profiles={"ml": ML, "medical": {"profile_version": "1.0", "dicom": {"Modality": "CT"}}},
        user={"note": [1, 2]},
        source=[{"format": "nifti", "header": {"descrip": "scan", "dim": [3, 30, 20, 10]}}],
        shape=SHAPE,
    )
    kwargs.update(overrides)
    return Meta(**kwargs)


class TestSpecExamples(unittest.TestCase):
    def test_spec_example_round_trips_through_dicts(self):
        meta = _meta()
        data = meta.to_dict()
        restored = Meta.from_dict(data, shape=SHAPE)
        self.assertEqual(restored.to_dict(), data)
        self.assertEqual(data["spaces"][0]["axes"][0], {"name": "x", "unit": "millimeter", "unit_code": "mm", "increases_toward": "left"})
        self.assertEqual(data["values"]["labels"]["1"]["code"]["value"], "10200004")

    def test_unknown_keys_are_kept_at_every_level(self):  # L9
        data = _meta().to_dict()
        data["pyramid"] = {"levels": []}
        data["axes"][0]["future"] = 1
        data["spaces"][0]["from_array"]["future"] = [1, 2]
        data["values"]["future"] = "x"
        data["profiles"]["org_example_cryoem"] = {"grid": 3}
        restored = Meta.from_dict(data, shape=SHAPE)
        self.assertEqual(restored.to_dict(), data)

    def test_minor_versions_are_read_and_major_versions_refused(self):
        data = _meta().to_dict()
        data["version"] = "1.7"
        self.assertEqual(Meta.from_dict(data, shape=SHAPE).version, "1.7")
        data["version"] = "2.0"
        with self.assertRaises(MetadataError):
            Meta.from_dict(data, shape=SHAPE)


class TestValueSemantics(unittest.TestCase):
    def test_equality_compares_values(self):  # H4
        self.assertEqual(_meta(), _meta())
        self.assertNotEqual(_meta(), _meta(values={"kind": "measurement"}))
        self.assertEqual(Axis.coerce("z"), Axis(name="z", type="space"))

    def test_copies_share_nothing(self):  # H5, H6
        meta = _meta()
        for clone in (meta.copy(), copy.deepcopy(meta), pickle.loads(pickle.dumps(meta))):
            clone.profiles["ml"]["split"] = "test"
            clone.user["note"].append(3)
            self.assertEqual(meta.profiles["ml"]["split"], "train")
            self.assertEqual(meta.user, {"note": [1, 2]})

    def test_inputs_are_copied(self):  # L8
        user = {"note": [1, 2]}
        profiles = {"ml": {"split": "train"}}
        meta = Meta(axes=["a"], user=user, profiles=profiles, shape=(4,))
        user["note"].append(3)
        profiles["ml"]["split"] = "test"
        self.assertEqual(meta.user, {"note": [1, 2]})
        self.assertEqual(meta.profiles["ml"]["split"], "train")


class TestValidation(unittest.TestCase):
    def test_failed_change_leaves_metadata_untouched(self):  # M7, L7
        meta = _meta()
        before = meta.to_dict()
        with self.assertRaises(MetadataError):
            meta.update(axes=["a", "b", "c"], spaces=[])  # boxes still refer to z, y, x
        with self.assertRaises(MetadataError):
            meta.axes = ["z", "y"]
        self.assertEqual(meta.to_dict(), before)

    def test_replacing_parts_is_validated_against_the_shape(self):  # H7, H8
        meta = _meta()
        with self.assertRaises(MetadataError):
            meta.axes = ["z", "y", "x", "t"]
        with self.assertRaises(MetadataError):
            meta.axes = [{"name": "z", "type": "channel", "labels": ["a", "b"]}, "y", "x"]
        with self.assertRaises(MetadataError):
            meta.spaces = [dict(PATIENT, from_array={"array_axes": ["z", "y", "q"], "scale": [1, 1, 1]})]
        meta.update(axes=[{"name": "c", "type": "channel", "labels": [str(i) for i in range(10)]}, "y", "x"],
                    spaces=[], profiles={"ml": {"split": "test"}})
        self.assertEqual(len(meta.axes), 3)

    def test_read_only_parts_cannot_be_mutated(self):  # M8
        meta = _meta()
        with self.assertRaises(TypeError):
            meta.source[0].header["descrip"] = "changed"
        with self.assertRaises((TypeError, AttributeError)):
            meta.source[0].header["dim"].append(1)
        with self.assertRaises(AttributeError):
            meta.axes[0].name = "q"
        with self.assertRaises(AttributeError):
            meta.source = []

    def test_numpy_scalars_are_converted_and_bools_rejected(self):  # M9
        values = Values.from_dict({"kind": "measurement", "scale": np.float32(1.5), "offset": np.int64(-1024)})
        self.assertIs(type(values.scale), float)
        self.assertEqual(values.offset, -1024.0)
        meta = Meta(axes=["a"], user={"n": np.int16(3), "a": np.arange(3)}, shape=(4,))
        self.assertEqual(meta.user, {"n": 3, "a": [0, 1, 2]})
        self.assertIs(type(meta.user["n"]), int)
        for bad in ({"kind": "measurement", "scale": True}, {"kind": "measurement", "fill": np.bool_(False)}):
            with self.assertRaises(MetadataError):
                Values.from_dict(bad)

    def test_user_dicts_are_never_unwrapped(self):  # M10
        meta = Meta(axes=["a"], user={"data": {"x": 1}}, shape=(4,))
        self.assertEqual(meta.user, {"data": {"x": 1}})
        meta.user = {"data": 5}
        self.assertEqual(meta.user, {"data": 5})

    def test_geometry_and_box_checks(self):  # M11
        bad_spaces = [
            dict(PATIENT, from_array=dict(PATIENT["from_array"], matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])),  # 3 space rows for 3 axes, missing last row
            dict(PATIENT, from_array=dict(PATIENT["from_array"], matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])),
            dict(PATIENT, from_array=dict(PATIENT["from_array"], matrix=[[float("nan"), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
            dict(PATIENT, from_array=dict(PATIENT["from_array"], matrix=[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
            dict(PATIENT, from_array={"array_axes": ["z", "y", "x"], "scale": [1, 0, 1]}),
            dict(PATIENT, from_array={"array_axes": ["z", "y", "x"], "scale": [1, 1]}),
            dict(PATIENT, from_array={"array_axes": ["z", "y", "x"], "matrix": PATIENT["from_array"]["matrix"], "scale": [1, 1, 1]}),
            dict(PATIENT, axes=PATIENT["axes"][:2]),
            dict(PATIENT, axes=[{"name": "x"}, {"name": "y", "unit": "millimeter"}, {"name": "z", "unit": "millimeter"}]),
        ]
        for i, space in enumerate(bad_spaces):
            with self.subTest(case=i), self.assertRaises(MetadataError):
                _meta(spaces=[space])
        bad_boxes = [
            {"frame": "index", "axes": ["z", "y"], "items": [{"min": [1, 2, 3], "max": [4, 5, 6]}]},
            {"frame": "index", "axes": ["z", "y", "x"], "items": [{"min": [5, 2, 3], "max": [4, 5, 6]}]},
            {"frame": "index", "axes": ["z", "y", "q"], "items": []},
            {"frame": "atlas", "axes": ["x"], "items": []},
        ]
        for i, boxes in enumerate(bad_boxes):
            with self.subTest(box=i), self.assertRaises(MetadataError):
                _meta(profiles={"ml": {"boxes": boxes}})
        with self.assertRaises(MetadataError):
            _meta(profiles={"ml": {"statistics": {"min": 5, "max": 1}}})
        boxes_in_space = {"frame": "patient", "axes": ["x", "y"], "items": [{"min": [-10.5, 0], "max": [3, 4.5]}]}
        self.assertEqual(_meta(profiles={"ml": {"boxes": boxes_in_space}}).profiles["ml"]["boxes"], boxes_in_space)

    def test_unserialisable_values_give_type_error(self):  # L10
        with self.assertRaisesRegex(TypeError, r"user\['bad'\]"):
            Meta(axes=["a"], user={"bad": object()}, shape=(4,))
        with self.assertRaisesRegex(TypeError, "profiles"):
            _meta(profiles={"x": {"f": lambda: 1}})

    def test_space_axes_have_no_sampling_and_labels_match_length(self):
        with self.assertRaises(MetadataError):
            Axis.from_dict({"name": "z", "type": "space", "step": 1.0})
        with self.assertRaises(MetadataError):
            Meta(axes=[{"name": "t", "type": "time", "coordinates": [0, 1]}], shape=(3,))
        meta = Meta(axes=[{"name": "t", "type": "time", "start": 0, "step": 2.5, "unit": "second"}], shape=(3,))
        self.assertEqual(meta.axes[0].unit_code, "s")

    def test_label_rules(self):
        with self.assertRaises(MetadataError):
            Values.from_dict({"kind": "measurement", "labels": {"1": {"name": "x"}}})
        with self.assertRaises(MetadataError):
            Values.from_dict({"kind": "label", "labels": {"one": {"name": "x"}}})
        with self.assertRaises(MetadataError):
            Values.from_dict({"kind": "label", "labels": {"1": {"color": [300, 0, 0]}}})
        with self.assertRaises(MetadataError):
            _meta(values={"kind": "probability", "label_layout": "channel"})
        self.assertEqual(Values.from_dict({"kind": "label", "labels": {1: {"name": "x"}}}).labels["1"]["name"], "x")
        self.assertEqual(
            Values.from_dict(LABELS).without_label_semantics(),
            Values(kind="measurement"),
        )

    def test_history_is_append_only(self):
        meta = _meta()
        meta.add_history("converted", tool="medvol 2.0.0", warnings=["w"])
        self.assertEqual(meta.history[0]["action"], "converted")
        self.assertTrue(meta.history[0]["time"].endswith("Z"))
        with self.assertRaises(TypeError):
            meta.history[0]["action"] = "x"


class TestGeometry(unittest.TestCase):
    def test_convenience_geometry_matches_spec_example(self):
        space = space_from_geometry(
            "patient", ["z", "y", "x"],
            spacing=[2.5, 0.7, 0.7], origin=[-170.1, -201.3, -87.0],
            direction=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        )
        np.testing.assert_allclose(space.matrix(), np.array(PATIENT["from_array"]["matrix"]))
        self.assertEqual(space.spacing, (2.5, 0.7, 0.7))
        self.assertEqual(space.origin, (-170.1, -201.3, -87.0))
        self.assertEqual(space.letter_codes(), ("I", "P", "L"))

    def test_shorthand_and_derived_values(self):
        space = Space.from_dict(dict(PATIENT, from_array={"array_axes": ["z", "y", "x"], "scale": [-2.0, 0.5, 0.5], "translation": [1, 2, 3]}))
        np.testing.assert_array_equal(space.matrix(), [[-2, 0, 0, 1], [0, 0.5, 0, 2], [0, 0, 0.5, 3], [0, 0, 0, 1]])
        self.assertEqual(space.spacing, (2.0, 0.5, 0.5))
        self.assertEqual(space.letter_codes(), ("R", "P", "S"))

    def test_shear_keeps_unit_direction_columns(self):
        matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.3, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        space = Space.from_dict(dict(PATIENT, from_array={"array_axes": ["z", "y", "x"], "matrix": matrix}))
        np.testing.assert_allclose(np.linalg.norm(space.direction, axis=0), 1.0)
        np.testing.assert_allclose(space.direction * np.array(space.spacing), np.array(matrix)[:3, :3])

    def test_geometry_input_checks(self):
        for kwargs in (
            dict(spacing=[1, 1]),
            dict(spacing=[1, -1, 1]),
            dict(direction=[[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
            dict(origin=[0, 0]),
        ):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(MetadataError):
                space_from_geometry("patient", ["z", "y", "x"], **kwargs)
        with self.assertRaises(MetadataError):
            space_from_geometry("stage", ["y", "x"], spacing=[1, 1])


class TestSourceAndStorage(unittest.TestCase):
    def test_source_entry_keeps_header_types_and_order(self):
        header = {"b": 1, "a": 0.7, "bin": b"\x00\x01", "seq": [{"k": "v"}], "(0029,1010)": {"vr": "OB", "value": b"x"}}
        entry = SourceEntry.from_dict({
            "format": "dicom", "header": header,
            "index_map": {"axis_order": [2, 1, 0], "flipped": [False, False, True], "offset": [0, 0, 0]},
        })
        self.assertEqual(list(entry.header), list(header))
        self.assertEqual(entry.to_dict()["header"], header)
        with self.assertRaises(MetadataError):
            SourceEntry.from_dict({"format": "nifti", "index_map": {"axis_order": [0, 0, 1]}})

    def test_recorded_compression_turns_back_into_blosc2_parameters(self):  # M12 rest
        import blosc2

        storage = Storage.from_dict({"compression": {"codec": "LZ4HC", "level": 5, "filters": ["SHUFFLE"]}})
        cparams = storage.to_cparams()
        self.assertEqual((cparams.codec, cparams.clevel, cparams.filters[0], cparams.nthreads),
                         (blosc2.Codec.LZ4HC, 5, blosc2.Filter.SHUFFLE, 4))
        self.assertEqual(storage.to_cparams(nthreads=2).nthreads, 2)
        no_filter = Storage.from_dict({"compression": {"codec": "ZSTD", "level": 3, "filters": ["NOFILTER"]}}).to_cparams()
        self.assertNotIn(blosc2.Filter.SHUFFLE, no_filter.filters)
        array = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        self.assertTrue(np.array_equal(blosc2.asarray(array, cparams=no_filter)[...], array))

    def test_storage_holds_only_portable_fields(self):
        storage = Storage.from_dict({"chunk_size": [1, 64, 64], "block_size": [1, 32, 32], "patch_size": [64, 64],
                                     "compression": {"codec": "LZ4HC", "level": 5, "filters": ["SHUFFLE"]}})
        self.assertEqual(storage.to_dict()["compression"], {"codec": "LZ4HC", "level": 5, "filters": ["SHUFFLE"]})


class TestFingerprint(unittest.TestCase):
    def test_fingerprint_detects_data_shape_and_geometry_changes(self):
        array = np.arange(np.prod(SHAPE), dtype=np.int16).reshape(SHAPE)
        meta = _meta()
        fp = fingerprint.compute(array, meta)
        self.assertEqual(fp["shape"], list(SHAPE))
        self.assertEqual(fp["dtype"], "int16")
        self.assertTrue(fp["data"].startswith("xxh3_128:"))
        self.assertTrue(fingerprint.matches(fp, array, meta))

        changed = array.copy()
        changed[0, 0, 0] += 1
        self.assertFalse(fingerprint.matches(fp, changed, meta))
        self.assertFalse(fingerprint.matches(fp, array.astype(np.int32), meta))
        moved = _meta(spaces=[dict(PATIENT, from_array=dict(PATIENT["from_array"], matrix=[[0, 0, 0.7, -170.0], [0, 0.7, 0, -201.3], [-2.5, 0, 0, -87.0], [0, 0, 0, 1]]))])
        self.assertFalse(fingerprint.matches(fp, array, moved))

    def test_geometry_hash_ignores_differences_below_accuracy(self):
        matrix = [[0, 0, 0.7, -170.1], [0, 0.7, 0, -201.3], [-2.5, 0, 0, -87.0], [0, 0, 0, 1]]
        nudged = [[0, 0, 0.7 * (1 + 1e-13), -170.1], [0, 0.7, 0, -201.3], [-2.5, 0, 0, -87.0], [0, 0, 0, 1]]
        a = _meta(spaces=[dict(PATIENT, from_array=dict(PATIENT["from_array"], matrix=matrix))])
        b = _meta(spaces=[dict(PATIENT, from_array=dict(PATIENT["from_array"], matrix=nudged))])
        self.assertEqual(fingerprint.geometry_hash(a), fingerprint.geometry_hash(b))

    def test_data_hash_is_the_same_for_numpy_and_chunked_reads(self):
        array = np.random.default_rng(0).random((7, 5, 3)).astype(np.float32)

        class Chunked:
            shape, dtype = array.shape, array.dtype

            def __getitem__(self, key):
                return array[key]

        original = fingerprint._STEP_NBYTES
        fingerprint._STEP_NBYTES = 40
        try:
            self.assertEqual(fingerprint.data_hash(Chunked()), fingerprint.data_hash(array))
        finally:
            fingerprint._STEP_NBYTES = original


if __name__ == "__main__":
    unittest.main()
