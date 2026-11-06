import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import blosc2
import numpy as np

import mlarray.mlarray as mlmod
from mlarray import MLArray, MLARRAY_DEFAULT_READ_AHEAD, set_read_ahead


class TestReadAhead(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "a.mla"
        MLArray(np.arange(4 * 32 * 32, dtype=np.float32).reshape(4, 32, 32)).save(self.path)
        self._flag = mlmod._read_ahead_configured
        mlmod._read_ahead_configured = False

    def tearDown(self):
        mlmod._read_ahead_configured = self._flag
        self._tmp.cleanup()

    def _open_with(self, env):
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch.object(blosc2, "set_prefetch", create=True, return_value=0) as sp:
            if "BLOSC_PREFETCH" not in env:
                os.environ.pop("BLOSC_PREFETCH", None)
            arr = MLArray.open(self.path)
            self.assertEqual(arr[1, 2, 3], 1 * 32 * 32 + 2 * 32 + 3)
            MLArray.open(self.path)
            return sp

    def test_first_open_turns_it_on(self):
        sp = self._open_with({})
        sp.assert_called_once_with(MLARRAY_DEFAULT_READ_AHEAD)

    def test_environment_variable_wins(self):
        sp = self._open_with({"BLOSC_PREFETCH": "8"})
        sp.assert_not_called()

    def test_user_setting_wins(self):
        with mock.patch.object(blosc2, "set_prefetch", create=True, return_value=4) as sp:
            self.assertEqual(set_read_ahead(0), 4)
            MLArray.open(self.path)
        sp.assert_called_once_with(0)

    def test_without_blosc2_support(self):
        class NoReadAhead:  # blosc2 without set_prefetch (before read-ahead existed)
            def __getattr__(self, name):
                if name == "set_prefetch":
                    raise AttributeError(name)
                return getattr(blosc2, name)

        with mock.patch.object(mlmod, "blosc2", NoReadAhead()):
            self.assertEqual(set_read_ahead(4), 0)
            mlmod._read_ahead_configured = False
            arr = MLArray.open(self.path)
            self.assertEqual(arr.shape, (4, 32, 32))

if __name__ == "__main__":
    unittest.main()
