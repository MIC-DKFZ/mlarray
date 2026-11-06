"""The only place where MLArray metadata is serialized and deserialized.

Metadata lives in the blosc2 vlmeta of an ``.mla`` file under the key
``"mlarray"``.
"""

import gzip
import pickle

from mlarray.meta import Meta
from mlarray.utils import is_serializable

VLMETA_KEY = "mlarray"


def encode(meta: Meta) -> bytes:
    """Serialize metadata into the bytes stored in vlmeta."""
    mapping = meta.to_mapping()
    if not is_serializable(mapping):
        raise RuntimeError("Metadata is not serializable.")
    return gzip.compress(pickle.dumps(mapping, protocol=pickle.HIGHEST_PROTOCOL))


def decode(data: bytes) -> Meta:
    """Deserialize metadata bytes read from vlmeta."""
    return Meta.from_mapping(pickle.loads(gzip.decompress(data)))


def read(vlmeta) -> Meta:
    """Read the metadata from a blosc2 vlmeta container."""
    return decode(vlmeta[VLMETA_KEY])


def write(vlmeta, data: bytes) -> None:
    """Write encoded metadata into a blosc2 vlmeta container."""
    vlmeta[VLMETA_KEY] = data
