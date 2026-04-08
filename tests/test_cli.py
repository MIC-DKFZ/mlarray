import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mlarray import MLArray
from mlarray.cli import (
    _header_to_source_meta,
    convert_from_mlarray,
    convert_image,
    convert_to_mlarray,
)
from mlarray.meta import Meta


class FakeMedVol:
    loaded_paths = []
    created_instances = []
    source_payloads = {}

    def __init__(
        self,
        source,
        *,
        affine=None,
        spacing=None,
        origin=None,
        direction=None,
        header=None,
        coordinate_system=None,
        backend=None,
        canonicalize=True,
        remove_obliqueness=False,
    ) -> None:
        self.backend = backend
        self.canonicalize = canonicalize
        self.remove_obliqueness = remove_obliqueness

        if isinstance(source, (str, Path)):
            payload = type(self).source_payloads.get(
                str(source),
                {
                    "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                    "affine": np.array(
                        [
                            [1.5, 0.0, 0.0, 10.0],
                            [0.0, 2.5, 0.0, 20.0],
                            [0.0, 0.0, 3.5, 30.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=float,
                    ),
                    "header": {"source_key": "source_val", "nested": {"a": 1}},
                    "coordinate_system": "RAS+",
                },
            )
            self.array = np.asarray(payload["array"])
            self.affine = np.asarray(payload["affine"], dtype=float)
            self.header = dict(payload.get("header", {}))
            self.coordinate_system = payload.get("coordinate_system")
            self._coordinate_system = self.coordinate_system
            self.backend = payload.get("backend", backend)
            self.saved_path = None
            type(self).loaded_paths.append(str(source))
            return

        self.array = np.asarray(source)
        if affine is not None:
            self.affine = np.asarray(affine, dtype=float)
        else:
            spacing = np.asarray(spacing, dtype=float)
            origin = np.asarray(origin, dtype=float)
            direction = np.asarray(direction, dtype=float)
            self.affine = np.eye(self.array.ndim + 1, dtype=float)
            self.affine[:-1, :-1] = direction @ np.diag(spacing)
            self.affine[:-1, -1] = origin
        self.header = {} if header is None else dict(header)
        self.coordinate_system = coordinate_system
        self._coordinate_system = coordinate_system
        self.saved_path = None
        type(self).created_instances.append(self)

    @property
    def spacing(self):
        return np.diag(self.affine[:-1, :-1])

    @property
    def origin(self):
        return self.affine[:-1, -1]

    @property
    def direction(self):
        scales = self.spacing
        return self.affine[:-1, :-1] / scales

    @property
    def ndim(self):
        return self.array.ndim

    def save(self, filepath, *, backend=None):
        if backend is not None:
            self.backend = backend
        self.saved_path = str(filepath)
        if str(filepath).endswith(".nrrd"):
            cs = self._coordinate_system
            if cs in {"RAS", "RAS+"}:
                self.header["space"] = "right-anterior-superior"
            elif cs in {"LPS", "LPS+"}:
                self.header["space"] = "left-posterior-superior"
            else:
                self.header.pop("space", None)


class TestCLIConversion(unittest.TestCase):
    def setUp(self):
        FakeMedVol.loaded_paths = []
        FakeMedVol.created_instances = []
        FakeMedVol.source_payloads = {}

    def test_convert_to_mlarray_from_nifti_copies_medvol_header_and_spatial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "converted.mla"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray("input_case.nii.gz", output_path)

            loaded = MLArray(output_path)
            self.assertEqual(loaded.meta.source.to_plain(), {"source_key": "source_val", "nested": {"a": 1}})
            self.assertEqual(
                loaded.affine,
                [[1.5, 0.0, 0.0, 10.0], [0.0, 2.5, 0.0, 20.0], [0.0, 0.0, 3.5, 30.0], [0.0, 0.0, 0.0, 1.0]],
            )
            self.assertEqual(loaded.meta._image_meta_format.to_plain(), "nifti")
            self.assertEqual(loaded.meta.spatial.coord_system, "RAS+")
            # XYZ axis labels are set so Slicer uses the affine as-is (identity permutation).
            self.assertEqual(loaded.meta.spatial.axis_labels, ["spatial_x", "spatial_y", "spatial_z"])

    def test_convert_to_mlarray_from_nrrd_uses_medvol_coord_system(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "converted-ras.mla"
            FakeMedVol.source_payloads["input_case.nrrd"] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "affine": np.diag([1.5, 2.5, 3.5, 1.0]),
                "header": {"space": "right-anterior-superior", "source_key": "source_val"},
                "coordinate_system": "RAS+",
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray("input_case.nrrd", output_path)

            loaded = MLArray(output_path)
            self.assertEqual(loaded.meta.spatial.coord_system, "RAS+")

    def test_convert_to_mlarray_from_nrrd_without_explicit_space_leaves_coord_system_unset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "converted-none.mla"
            FakeMedVol.source_payloads["input_case.nrrd"] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "affine": np.diag([1.5, 2.5, 3.5, 1.0]),
                "header": {"source_key": "source_val"},
                "coordinate_system": None,
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray("input_case.nrrd", output_path)

            loaded = MLArray(output_path)
            self.assertIsNone(loaded.meta.spatial.coord_system)

    def test_convert_from_mlarray_to_nrrd_uses_only_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            image = MLArray(
                array,
                spacing=(0.7, 0.8, 0.9),
                origin=(1.0, 2.0, 3.0),
                direction=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                meta=Meta(
                    source={"keep": "me"},
                    extra={"ignore": "me"},
                    is_seg=True,
                ),
            )
            image.save(source_path)
            output_path = Path(tmpdir) / "exported.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            self.assertEqual(len(FakeMedVol.created_instances), 1)
            exported = FakeMedVol.created_instances[0]
            self.assertTrue(np.array_equal(exported.array, array))
            # No coord_system stored → treated as RAS+; extra/is_seg not in header.
            self.assertEqual(exported.header, {"keep": "me", "space": "right-anterior-superior"})
            self.assertEqual(exported.backend, "pynrrd")
            self.assertEqual(exported.saved_path, str(output_path))

    def test_convert_from_mlarray_to_nrrd_writes_ras_plus_space_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source-ras.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            MLArray(
                array,
                spacing=(0.7, 0.8, 0.9),
                origin=(1.0, 2.0, 3.0),
                direction=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                meta=Meta(
                    source={"keep": "me", "space": "left-posterior-superior", "NRRD_space": "left-posterior-superior"},
                    spatial={"coord_system": "RAS+"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported-ras.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertEqual(exported.header["space"], "right-anterior-superior")
            self.assertNotIn("NRRD_space", exported.header)

    def test_convert_from_mlarray_to_nrrd_writes_lps_plus_space_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source-lps.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            MLArray(
                array,
                spacing=(0.7, 0.8, 0.9),
                origin=(1.0, 2.0, 3.0),
                direction=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                meta=Meta(
                    source={"keep": "me"},
                    spatial={"coord_system": "LPS+"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported-lps.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertEqual(exported.header["space"], "left-posterior-superior")

    def test_convert_from_mlarray_to_nrrd_legacy_lps_value_maps_to_lps_plus_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source-legacy-lps.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            MLArray(
                array,
                spacing=(0.7, 0.8, 0.9),
                origin=(1.0, 2.0, 3.0),
                direction=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                meta=Meta(
                    source={"keep": "me"},
                    spatial={"coord_system": "LPS"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported-legacy-lps.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertEqual(exported.header["space"], "left-posterior-superior")

    def test_convert_from_mlarray_to_nrrd_treats_custom_coord_system_as_ras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source-custom.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            MLArray(
                array,
                spacing=(0.7, 0.8, 0.9),
                origin=(1.0, 2.0, 3.0),
                direction=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                meta=Meta(
                    source={"keep": "me", "space": "left-posterior-superior"},
                    spatial={"coord_system": "my_custom_frame"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported-custom.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            # Custom coord_system → assumed RAS+ (data was canonicalised on import).
            self.assertEqual(exported.header, {"keep": "me", "space": "right-anterior-superior"})

    def test_convert_from_mlarray_to_nrrd_treats_unknown_coord_system_as_ras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source-unknown.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            MLArray(
                array,
                spacing=(0.7, 0.8, 0.9),
                origin=(1.0, 2.0, 3.0),
                direction=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                meta=Meta(
                    source={"keep": "me", "space": "left-posterior-superior"},
                    spatial={"coord_system": "unknown"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported-unknown.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            # "unknown" coord_system → assumed RAS+ (data was canonicalised on import).
            self.assertEqual(exported.header, {"keep": "me", "space": "right-anterior-superior"})

    def test_convert_from_mlarray_with_affine_derives_spatial_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "affine_source.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            affine = [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            MLArray(array, affine=affine, meta={"keep": "me"}).save(source_path)
            output_path = Path(tmpdir) / "exported.nii.gz"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertTrue(np.array_equal(exported.affine, np.array(affine)))
            # NIfTI: backend=None so MedVol auto-selects nibabel
            self.assertIsNone(exported.backend)

    def test_convert_from_mlarray_to_nifti_does_not_add_coord_system_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source-nifti.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            MLArray(
                array,
                affine=[
                    [2.0, 0.0, 0.0, 10.0],
                    [0.0, 3.0, 0.0, 20.0],
                    [0.0, 0.0, 4.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                meta=Meta(
                    source={"keep": "me"},
                    spatial={"coord_system": "RAS+"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported.nii.gz"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertEqual(exported.header, {"keep": "me"})
            # NIfTI: backend=None so MedVol auto-selects nibabel
            self.assertIsNone(exported.backend)

    def test_convert_roundtrip_nifti_sets_ras_plus_on_import(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = "input_case.nii.gz"
            mla_path = Path(tmpdir) / "roundtrip-nifti.mla"
            output_path = Path(tmpdir) / "roundtrip.nii.gz"
            FakeMedVol.source_payloads[input_path] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "affine": np.diag([1.5, 2.5, 3.5, 1.0]),
                "header": {"keep": "me"},
                "coordinate_system": "RAS+",
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray(input_path, mla_path)
                convert_from_mlarray(mla_path, output_path)

            loaded = MLArray(mla_path)
            exported = FakeMedVol.created_instances[0]
            self.assertEqual(loaded.meta.spatial.coord_system, "RAS+")
            self.assertEqual(exported.header, {"keep": "me"})

    def test_convert_roundtrip_nrrd_ras_plus_preserves_representable_coord_system(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = "input_case.nrrd"
            mla_path = Path(tmpdir) / "roundtrip.mla"
            output_path = Path(tmpdir) / "roundtrip.nrrd"
            FakeMedVol.source_payloads[input_path] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "affine": np.diag([1.5, 2.5, 3.5, 1.0]),
                "header": {"space": "right-anterior-superior", "keep": "me"},
                "coordinate_system": "RAS+",
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray(input_path, mla_path)
                convert_from_mlarray(mla_path, output_path)

            loaded = MLArray(mla_path)
            exported = FakeMedVol.created_instances[0]
            self.assertEqual(loaded.meta.spatial.coord_system, "RAS+")
            self.assertEqual(exported.header["space"], "right-anterior-superior")

    def test_header_to_source_meta_converts_numpy_values(self):
        header = {
            "sizes": np.array([2, 3, 4], dtype=np.int64),
            "spacing": np.float32(1.5),
        }

        source = _header_to_source_meta(header)

        self.assertEqual(source, {"sizes": [2, 3, 4], "spacing": 1.5})

    def test_convert_image_rejects_unsupported_direction(self):
        with self.assertRaises(RuntimeError):
            convert_image("input.nii.gz", "output.nrrd")


if __name__ == "__main__":
    unittest.main()
