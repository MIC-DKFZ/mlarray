import math
import unittest
import warnings

import blosc2
import numpy as np

from mlarray import MLArray
from mlarray.blosc2_layout_strategies import comp_blosc2_params, select_blosc2_filter
from nnunet_reference import _nnunet_comp_blosc2_params

# Reference values produced by nnU-Net's own comp_blosc2_params
# (nnunetv2/training/dataloading/nnunet_dataset.py, commit ded2aa3a) for the
# equivalent (c, x, y, z) input, mapped back to the MLArray axes. nnU-Net's 2D
# patches on 3D images are written as (1, y, x).
# (image_size, spatial_axis_mask, patch_size, bytes_per_pixel, kwargs, chunk_size, block_size)
REFERENCE_LAYOUTS = [
    ((128, 256, 256), None, (128, 128, 128), 4, {}, [96, 128, 128], [32, 32, 32]),
    ((64, 512, 512), None, (40, 224, 192), 4, {}, [40, 192, 192], [8, 64, 64]),
    ((300, 512, 512), None, (64, 160, 160), 2, {}, [80, 192, 192], [16, 64, 64]),
    ((40, 40, 40), None, (128, 128, 128), 4, {}, [40, 40, 40], [32, 32, 32]),
    ((512, 512, 512), None, (192, 192, 192), 1, {}, [192, 192, 160], [64, 64, 32]),
    ((17, 333, 251), None, (16, 320, 256), 4, {}, [17, 333, 251], [4, 128, 64]),
    ((1000, 64, 64), None, (128, 64, 64), 4, {}, [192, 64, 64], [64, 32, 16]),
    ((96, 256, 256), None, (96, 96, 96), 8, {}, [64, 128, 96], [16, 32, 32]),
    ((200, 512, 512), None, (1, 512, 512), 4, {}, [1, 512, 512], [1, 256, 128]),
    ((200, 512, 512), None, (1, 256, 256), 2, {}, [1, 384, 384], [1, 256, 256]),
    ((96, 256, 256), None, (1, 128, 128), 1, {}, [1, 192, 192], [1, 128, 128]),
    ((1024, 1024), None, (512, 512), 4, {}, [768, 768], [256, 128]),
    ((300, 4000), None, (256, 256), 1, {}, [300, 384], [256, 256]),
    ((4, 128, 160, 112), [False, True, True, True], (128, 128, 128), 4, {}, [1, 128, 128, 96], [1, 32, 32, 32]),
    ((128, 160, 112, 3), [True, True, True, False], (96, 96, 96), 4, {}, [128, 128, 96, 1], [32, 32, 32, 1]),
    ((3, 512, 512), [False, True, True], (256, 256), 1, {}, [1, 384, 384], [1, 256, 256]),
    ((2, 200, 512, 512), [False, True, True, True], (1, 384, 384), 2, {}, [1, 1, 512, 512], [1, 1, 256, 256]),
    ((256, 256, 256), None, (64, 64, 64), 4, {"max_chunk_to_patch_ratio_per_axis": None}, [128, 128, 96], [32, 32, 32]),
    ((256, 256, 256), None, (64, 64, 64), 4, {"max_block_nbytes": 512 * 1024, "max_chunk_nbytes": 16 * 1024**2}, [96, 96, 96], [64, 64, 32]),
    ((64, 256, 256), None, (1, 128, 128), 4, {"grow_singleton_patch_axes": True}, [1, 192, 192], [1, 128, 128]),
]


def _nnunet_layout(shape, patch, mask, bpp, kwargs):
    """nnU-Net's verbatim function on the equivalent (c, x, y, z) input, mapped back."""
    spatial = [i for i, s in enumerate(mask) if s]
    image = (1, *[shape[i] for i in spatial]) if len(spatial) == 3 else (1, 1, *[shape[i] for i in spatial])
    if len(spatial) == 3 and patch[0] == 1 and not kwargs.get("grow_singleton_patch_axes"):
        patch = patch[1:]  # nnU-Net's 2D configuration
    block, chunk = _nnunet_comp_blosc2_params(image, patch, bytes_per_pixel=bpp, **kwargs)
    chunk_size, block_size = [1] * len(shape), [1] * len(shape)
    for axis, c, b in zip(spatial, chunk[-len(spatial):], block[-len(spatial):]):
        chunk_size[axis], block_size[axis] = c, b
    return chunk_size, block_size


def _random_case(rng):
    n_spatial = int(rng.integers(2, 4))
    spatial = [int(rng.integers(1, 700)) for _ in range(n_spatial)]
    patch = [int(rng.integers(1, 400)) for _ in range(n_spatial)]
    if n_spatial == 3 and rng.random() < 0.3:
        patch[0] = 1
    shape, mask = spatial.copy(), [True] * n_spatial
    for _ in range(int(rng.integers(0, 3))):
        pos = int(rng.integers(0, len(shape) + 1))
        shape.insert(pos, int(rng.integers(1, 6)))
        mask.insert(pos, False)
    kwargs = {}
    if rng.random() < 0.3:
        kwargs["max_chunk_nbytes"] = int(rng.choice([1572864, 6 * 1024**2, 24 * 1024**2]))
    if rng.random() < 0.3:
        kwargs["max_block_nbytes"] = int(rng.choice([32768, 131072, 524288]))
    if rng.random() < 0.2:
        kwargs["max_chunk_to_patch_ratio_per_axis"] = None
    return tuple(shape), tuple(patch), mask, int(rng.choice([1, 2, 4, 8])), kwargs


def _quantized(seed=0):
    """z-scored integer data: few distinct values, like non-resampled images."""
    rng = np.random.default_rng(seed)
    v = np.cumsum(rng.integers(-3, 4, (64, 96, 96)), axis=2).astype(np.int16)
    return ((v - v.mean()) / v.std()).astype(np.float32)


def _interpolated(seed=0):
    """Linearly interpolated smooth data, like resampled images."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((17, 25, 25)).astype(np.float32)
    for axis, n in enumerate((64, 96, 96)):
        pos = np.linspace(0, a.shape[axis] - 1, n)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, a.shape[axis] - 1)
        shape = [1] * a.ndim
        shape[axis] = -1
        w = (pos - i0).astype(np.float32).reshape(shape)
        a = a.take(i0, axis) * (1 - w) + a.take(i1, axis) * w
    return (a + rng.standard_normal(a.shape).astype(np.float32) * 0.01).astype(np.float32)


class TestLayout(unittest.TestCase):
    def test_matches_nnunet_reference_values(self):
        for image, mask, patch, bpp, kwargs, chunk, block in REFERENCE_LAYOUTS:
            with self.subTest(image=image, mask=mask, patch=patch, bpp=bpp, kwargs=kwargs):
                self.assertEqual(
                    comp_blosc2_params(image, patch, spatial_axis_mask=mask, bytes_per_pixel=bpp, **kwargs),
                    (chunk, block),
                )

    def test_equals_nnunet_for_2_and_3_spatial_axes(self):
        rng = np.random.default_rng(0)
        for _ in range(3000):
            shape, patch, mask, bpp, kwargs = _random_case(rng)
            with self.subTest(shape=shape, patch=patch, mask=mask, bpp=bpp, kwargs=kwargs):
                self.assertEqual(
                    comp_blosc2_params(shape, patch, mask, bytes_per_pixel=bpp, **kwargs),
                    _nnunet_layout(shape, patch, mask, bpp, kwargs),
                )

    def test_any_number_of_spatial_axes(self):
        for shape, patch in [((100000,), (4096,)), ((40, 50, 60, 70), (16, 32, 32, 32)),
                             ((6, 7, 8, 9, 10), (4, 4, 4, 4, 4))]:
            with self.subTest(shape=shape, patch=patch):
                chunk, block = comp_blosc2_params(shape, patch, bytes_per_pixel=4)
                self.assertTrue(all(1 <= b <= c <= s for b, c, s in zip(block, chunk, shape)))
                self.assertLessEqual(math.prod(block) * 4, 128 * 1024)
                self.assertLessEqual(math.prod(chunk) * 4, 6 * 1024**2)

    def test_non_spatial_axes_get_size_one(self):
        chunk, block = comp_blosc2_params((2, 5, 64, 64, 64), (32, 32, 32), [False, False, True, True, True])
        self.assertEqual((chunk[:2], block[:2]), ([1, 1], [1, 1]))

    def test_invalid_inputs(self):
        with self.assertRaises(RuntimeError):
            comp_blosc2_params((64, 64), (32, 32, 32))
        with self.assertRaises(RuntimeError):
            comp_blosc2_params((64, 64, 64), (32, 32))
        with self.assertRaises(RuntimeError):
            comp_blosc2_params((4, 64), (32,), [False, False])
        with self.assertRaises(RuntimeError):
            comp_blosc2_params((64, 64, 64), (32, 32, 32), spatial_axis_mask=[True, True])


class TestFilterSelection(unittest.TestCase):
    # Reference filters from nnU-Net's _select_filter (commit ded2aa3a).
    def test_matches_nnunet_reference(self):
        cases = [
            (_quantized, (32, 96, 96), (8, 32, 32), blosc2.Filter.NOFILTER),
            (_quantized, (64, 96, 96), (16, 32, 32), blosc2.Filter.NOFILTER),
            (_interpolated, (32, 96, 96), (8, 32, 32), blosc2.Filter.SHUFFLE),
            (_interpolated, (64, 96, 96), (16, 32, 32), blosc2.Filter.SHUFFLE),
        ]
        for make, chunk, block, expected in cases:
            with self.subTest(data=make.__name__, chunk=chunk):
                self.assertEqual(select_blosc2_filter(make(), chunk, block), expected)

    def test_falls_back_to_nofilter(self):
        with self.assertWarns(UserWarning):
            f = select_blosc2_filter(np.zeros((4, 4), np.float32), (0, 4), (1, 4))
        self.assertEqual(f, blosc2.Filter.NOFILTER)


class TestMLArrayLayout(unittest.TestCase):
    def test_default_layout_is_nnunet(self):
        data = np.zeros((300, 64, 64), dtype=np.uint8)
        arr = MLArray(data, patch_size=(64, 32, 32))
        self.assertEqual(
            (list(arr.meta.storage.chunk_size), list(arr.meta.storage.block_size)),
            comp_blosc2_params(data.shape, (64, 32, 32), bytes_per_pixel=1),
        )

    def test_comp_blosc2_params_classmethod(self):
        self.assertEqual(
            MLArray.comp_blosc2_params((128, 256, 256), (128, 128, 128)),
            ([96, 128, 128], [32, 32, 32]),
        )


if __name__ == "__main__":
    unittest.main()


class TestWriteDefaults(unittest.TestCase):
    def _saved(self, data, **kwargs):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.mla"
            MLArray(data, **kwargs).save(path)
            store = blosc2.open(str(path), mode="r")
            return MLArray(path), store.schunk.cparams

    def test_lz4hc_5_with_4_threads(self):
        arr, cparams = self._saved(_interpolated(), patch_size=(32, 48, 48))
        self.assertEqual((cparams.codec, cparams.clevel), (blosc2.Codec.LZ4HC, 5))
        meta = arr.meta.storage.compression
        self.assertEqual((blosc2.Codec[meta["codec"]], meta["level"]), (blosc2.Codec.LZ4HC, 5))
        self.assertEqual(MLArray(_interpolated(), patch_size=(32, 48, 48))._store.schunk.cparams.nthreads, 4)

    def test_filter_chosen_from_data(self):
        for make, expected in ((_quantized, blosc2.Filter.NOFILTER), (_interpolated, blosc2.Filter.SHUFFLE)):
            with self.subTest(data=make.__name__):
                arr, cparams = self._saved(make(), patch_size=(32, 48, 48))
                self.assertEqual(blosc2.Filter.SHUFFLE in cparams.filters, expected == blosc2.Filter.SHUFFLE)
                self.assertEqual(blosc2.Filter[arr.meta.storage.compression["filters"][0]], expected)
                self.assertTrue(np.array_equal(arr[...], make()))

    def test_explicit_cparams_are_kept(self):
        cp = {"codec": blosc2.Codec.ZSTD, "clevel": 3}
        arr, cparams = self._saved(_quantized(), patch_size=(32, 48, 48), cparams=cp)
        self.assertEqual((cparams.codec, cparams.clevel), (blosc2.Codec.ZSTD, 3))
        self.assertNotIn("filters", arr.meta.storage.compression)
