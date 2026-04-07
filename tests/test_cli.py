import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mlarray import MLArray
from mlarray.cli import convert_from_mlarray, convert_image, convert_to_mlarray
from mlarray.meta import Meta


class FakeMedVol:
    loaded_paths = []
    created_instances = []
    source_payloads = {}

    def __init__(
        self,
        array,
        spacing=None,
        origin=None,
        direction=None,
        header=None,
        is_seg=None,
        copy=None,
    ) -> None:
        if isinstance(array, (str, Path)):
            payload = type(self).source_payloads.get(
                str(array),
                {
                    "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                    "spacing": np.array([1.5, 2.5, 3.5], dtype=float),
                    "origin": np.array([10.0, 20.0, 30.0], dtype=float),
                    "direction": np.array(
                        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                        dtype=float,
                    ),
                    "header": {"source_key": "source_val", "nested": {"a": 1}},
                },
            )
            self.array = np.asarray(payload["array"])
            self.spacing = np.asarray(payload["spacing"], dtype=float)
            self.origin = np.asarray(payload["origin"], dtype=float)
            self.direction = np.asarray(payload["direction"], dtype=float)
            self.header = dict(payload.get("header", {}))
            self.is_seg = None
            self.saved_path = None
            type(self).loaded_paths.append(str(array))
            return

        self.array = np.asarray(array)
        self.spacing = None if spacing is None else np.asarray(spacing, dtype=float)
        self.origin = None if origin is None else np.asarray(origin, dtype=float)
        self.direction = None if direction is None else np.asarray(direction, dtype=float)
        self.header = {} if header is None else dict(header)
        self.is_seg = is_seg
        self.saved_path = None
        type(self).created_instances.append(self)

    def save(self, filepath):
        self.saved_path = str(filepath)


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
            self.assertEqual(loaded.spacing, [1.5, 2.5, 3.5])
            self.assertEqual(loaded.origin, [10.0, 20.0, 30.0])
            self.assertEqual(
                loaded.direction,
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            )
            self.assertEqual(loaded.meta._image_meta_format.to_plain(), "nifti")
            self.assertEqual(loaded.meta.spatial.coord_system, "LPS")

    def test_convert_to_mlarray_from_nrrd_sets_coord_system_ras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "converted-ras.mla"
            FakeMedVol.source_payloads["input_case.nrrd"] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "spacing": np.array([1.5, 2.5, 3.5], dtype=float),
                "origin": np.array([10.0, 20.0, 30.0], dtype=float),
                "direction": np.eye(3, dtype=float),
                "header": {"space": "right-anterior-superior", "source_key": "source_val"},
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray("input_case.nrrd", output_path)

            loaded = MLArray(output_path)
            self.assertEqual(loaded.meta.spatial.coord_system, "RAS")

    def test_convert_to_mlarray_from_nrrd_sets_coord_system_lps_from_nrrd_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "converted-lps.mla"
            FakeMedVol.source_payloads["input_case.nrrd"] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "spacing": np.array([1.5, 2.5, 3.5], dtype=float),
                "origin": np.array([10.0, 20.0, 30.0], dtype=float),
                "direction": np.eye(3, dtype=float),
                "header": {"NRRD_space": "left-posterior-superior"},
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray("input_case.nrrd", output_path)

            loaded = MLArray(output_path)
            self.assertEqual(loaded.meta.spatial.coord_system, "LPS")

    def test_convert_to_mlarray_from_nrrd_preserves_other_explicit_space_strings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "converted-scanner-xyz.mla"
            FakeMedVol.source_payloads["input_case.nrrd"] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "spacing": np.array([1.5, 2.5, 3.5], dtype=float),
                "origin": np.array([10.0, 20.0, 30.0], dtype=float),
                "direction": np.eye(3, dtype=float),
                "header": {"space": "scanner-xyz"},
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray("input_case.nrrd", output_path)

            loaded = MLArray(output_path)
            self.assertEqual(loaded.meta.spatial.coord_system, "scanner-xyz")

    def test_convert_to_mlarray_from_nrrd_without_explicit_space_leaves_coord_system_unset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "converted-none.mla"
            FakeMedVol.source_payloads["input_case.nrrd"] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "spacing": np.array([1.5, 2.5, 3.5], dtype=float),
                "origin": np.array([10.0, 20.0, 30.0], dtype=float),
                "direction": np.eye(3, dtype=float),
                "header": {"source_key": "source_val"},
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
            self.assertEqual(exported.header, {"keep": "me"})
            self.assertEqual(exported.is_seg, None)
            self.assertTrue(np.array_equal(exported.spacing, np.array([0.7, 0.8, 0.9])))
            self.assertTrue(np.array_equal(exported.origin, np.array([1.0, 2.0, 3.0])))
            self.assertTrue(
                np.array_equal(
                    exported.direction,
                    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                )
            )
            self.assertEqual(exported.saved_path, str(output_path))

    def test_convert_from_mlarray_to_nrrd_writes_ras_space_metadata(self):
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
                    spatial={"coord_system": "RAS"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported-ras.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertEqual(exported.header["space"], "right-anterior-superior")
            self.assertNotIn("NRRD_space", exported.header)

    def test_convert_from_mlarray_to_nrrd_writes_lps_space_metadata(self):
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
                    spatial={"coord_system": "LPS"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported-lps.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertEqual(exported.header["space"], "left-posterior-superior")

    def test_convert_from_mlarray_to_nrrd_preserves_supported_explicit_space_strings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source-scanner-xyz.mla"
            array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            MLArray(
                array,
                spacing=(0.7, 0.8, 0.9),
                origin=(1.0, 2.0, 3.0),
                direction=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                meta=Meta(
                    source={"keep": "me"},
                    spatial={"coord_system": "scanner-xyz"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported-scanner-xyz.nrrd"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertEqual(exported.header["space"], "scanner-xyz")

    def test_convert_from_mlarray_to_nrrd_drops_unsupported_coord_system_strings(self):
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
            self.assertEqual(exported.header, {"keep": "me"})

    def test_convert_from_mlarray_to_nrrd_drops_space_when_coord_system_unknown(self):
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
            self.assertEqual(exported.header, {"keep": "me"})

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
            self.assertTrue(np.array_equal(exported.spacing, np.array([2.0, 3.0, 4.0])))
            self.assertTrue(np.array_equal(exported.origin, np.array([10.0, 20.0, 30.0])))
            self.assertTrue(np.array_equal(exported.direction, np.eye(3)))

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
                    spatial={"coord_system": "RAS"},
                ),
            ).save(source_path)
            output_path = Path(tmpdir) / "exported.nii.gz"

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_from_mlarray(source_path, output_path)

            exported = FakeMedVol.created_instances[0]
            self.assertEqual(exported.header, {"keep": "me"})
            self.assertFalse(hasattr(exported, "coord_system"))

    def test_convert_roundtrip_nifti_sets_lps_on_import_but_does_not_export_explicit_coord_system(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = "input_case.nii.gz"
            mla_path = Path(tmpdir) / "roundtrip-nifti.mla"
            output_path = Path(tmpdir) / "roundtrip.nii.gz"
            FakeMedVol.source_payloads[input_path] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "spacing": np.array([1.5, 2.5, 3.5], dtype=float),
                "origin": np.array([10.0, 20.0, 30.0], dtype=float),
                "direction": np.eye(3, dtype=float),
                "header": {"keep": "me"},
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray(input_path, mla_path)
                convert_from_mlarray(mla_path, output_path)

            loaded = MLArray(mla_path)
            exported = FakeMedVol.created_instances[0]
            self.assertEqual(loaded.meta.spatial.coord_system, "LPS")
            self.assertEqual(exported.header, {"keep": "me"})

    def test_convert_roundtrip_nrrd_ras_preserves_representable_coord_system(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = "input_case.nrrd"
            mla_path = Path(tmpdir) / "roundtrip.mla"
            output_path = Path(tmpdir) / "roundtrip.nrrd"
            FakeMedVol.source_payloads[input_path] = {
                "array": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                "spacing": np.array([1.5, 2.5, 3.5], dtype=float),
                "origin": np.array([10.0, 20.0, 30.0], dtype=float),
                "direction": np.eye(3, dtype=float),
                "header": {"space": "right-anterior-superior", "keep": "me"},
            }

            with patch("mlarray.cli.MedVol", FakeMedVol):
                convert_to_mlarray(input_path, mla_path)
                convert_from_mlarray(mla_path, output_path)

            loaded = MLArray(mla_path)
            exported = FakeMedVol.created_instances[0]
            self.assertEqual(loaded.meta.spatial.coord_system, "RAS")
            self.assertEqual(exported.header["space"], "right-anterior-superior")

    def test_convert_image_rejects_unsupported_direction(self):
        with self.assertRaises(RuntimeError):
            convert_image("input.nii.gz", "output.nrrd")


if __name__ == "__main__":
    unittest.main()
