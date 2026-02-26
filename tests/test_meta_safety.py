import unittest

import numpy as np

from mlarray import MLArray
from mlarray.meta import Meta, MetaBlosc2, MetaBbox, MetaExtra, MetaSource, MetaSpatial, _meta_internal_write


class TestMetaSafety(unittest.TestCase):
    def test_mlarray_meta_is_read_only_attribute(self):
        image = MLArray(np.zeros((4, 4, 4), dtype=np.float32))
        with self.assertRaises(AttributeError):
            image.meta = Meta()

    def test_meta_namespace_assignment_is_coerced(self):
        image = MLArray(np.zeros((4, 4, 4), dtype=np.float32))
        image.meta.source = {"a": 1}
        image.meta.extra = {"b": 2}
        image.meta.bbox = {"bboxes": [[[0, 1], [2, 3], [4, 5]]]}
        image.meta.is_seg = True

        self.assertTrue(isinstance(image.meta.source, MetaSource))
        self.assertTrue(isinstance(image.meta.extra, MetaExtra))
        self.assertTrue(isinstance(image.meta.bbox, MetaBbox))
        self.assertEqual(image.meta.source["a"], 1)
        self.assertEqual(image.meta.extra["b"], 2)
        self.assertTrue(image.meta.is_seg)

    def test_internal_meta_fields_are_read_only_for_users(self):
        image = MLArray(np.zeros((4, 4, 4), dtype=np.float32))

        with self.assertRaises(AttributeError):
            image.meta._has_array.has_array = False
        with self.assertRaises(AttributeError):
            image.meta._image_meta_format = "nifti"
        with self.assertRaises(AttributeError):
            image.meta._mlarray_version = "v999"
        with self.assertRaises(AttributeError):
            image.meta.spatial.shape = [1, 1, 1]
        with self.assertRaises(AttributeError):
            image.meta.spatial.shape[0] = 1
        with self.assertRaises(AttributeError):
            image.meta.blosc2.chunk_size = [1, 1, 1]
        with self.assertRaises(AttributeError):
            image.meta.blosc2.chunk_size[0] = 1

    def test_spatial_assignment_revalidates_using_cached_dims(self):
        image = MLArray(np.zeros((4, 4, 4), dtype=np.float32))
        image.meta.spatial.spacing = [0.1, 0.2, 0.3]
        self.assertEqual(image.meta.spatial.spacing, [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            image.meta.spatial.spacing = [0.1, 0.2]

    def test_internal_cached_validation_state_is_hidden(self):
        image = MLArray(np.zeros((4, 4, 4), dtype=np.float32))
        mapping = image.meta.spatial.to_mapping(include_none=True)
        plain = image.meta.spatial.to_plain(include_none=True)
        self.assertNotIn("_validate_ndims", mapping)
        self.assertNotIn("_validate_spatial_ndims", mapping)
        self.assertNotIn("_validate_ndims", plain)
        self.assertNotIn("_validate_spatial_ndims", plain)
        self.assertNotIn("_validate_ndims", repr(image.meta.spatial))
        self.assertNotIn("_validate_spatial_ndims", str(image.meta.spatial))

    def test_meta_spatial_rejects_affine_mix_with_spacing_origin_direction(self):
        affine = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        with self.assertRaises(ValueError):
            MetaSpatial(
                spacing=[1.0, 1.0, 1.0],
                origin=[0.0, 0.0, 0.0],
                direction=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                affine=affine,
            )

    def test_meta_setters_accept_raw_values(self):
        meta = Meta()
        meta.set_source({"patient_id": "p-001"})
        meta.set_extra({"pipeline": "v1"})
        meta.set_is_seg(True)
        meta.set_bbox([[[0, 1], [2, 3], [4, 5]]])
        meta.add_bbox([[1, 2], [3, 4], [5, 6]])
        meta.update_extra({"stage": "train"})
        meta.set_stats({"min": 0.0, "max": 1.0, "mean": 0.5})
        meta.set_spatial(
            {
                "spacing": [1.0, 2.0, 3.0],
                "origin": [0.0, 0.0, 0.0],
                "direction": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        )

        self.assertEqual(meta.source["patient_id"], "p-001")
        self.assertEqual(meta.extra["pipeline"], "v1")
        self.assertEqual(meta.extra["stage"], "train")
        self.assertTrue(meta.is_seg)
        self.assertEqual(len(meta.bbox.bboxes), 2)
        self.assertIsNone(meta.bbox.scores)
        self.assertIsNone(meta.bbox.labels)
        self.assertEqual(meta.stats.mean, 0.5)
        self.assertEqual(meta.spatial.spacing, [1.0, 2.0, 3.0])

    def test_user_copy_from_skips_internal_fields(self):
        dst = Meta()
        src = Meta(source={"a": 1})
        with _meta_internal_write():
            src._has_array.has_array = True
            src.blosc2 = MetaBlosc2(
                chunk_size=[1, 2, 3],
                block_size=[1, 1, 1],
                patch_size=[8, 8, 8],
            )

        dst.copy_from(src, overwrite=True)

        self.assertEqual(dst.source["a"], 1)
        self.assertFalse(dst._has_array.has_array)
        self.assertIsNone(dst.blosc2.chunk_size)

    def test_internal_copy_from_can_copy_internal_fields(self):
        dst = Meta()
        src = Meta(source={"a": 1})
        with _meta_internal_write():
            src._has_array.has_array = True
            src.blosc2 = MetaBlosc2(
                chunk_size=[1, 2, 3],
                block_size=[1, 1, 1],
                patch_size=[8, 8, 8],
            )
            dst.copy_from(src, overwrite=True)

        self.assertTrue(dst._has_array.has_array)
        self.assertEqual(dst.blosc2.chunk_size, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
