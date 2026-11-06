import json
import unittest
from importlib import resources

import jsonschema
from referencing import Registry, Resource

from test_metadata_model import _meta

SCHEMA_FILES = ("mlarray-1.0.json", "profile-ml-1.0.json", "profile-medical-1.0.json")


def _load(name):
    return json.loads(resources.files("mlarray").joinpath(f"schemas/{name}").read_text())


def _validator(ref="mlarray-1.0.json"):
    schemas = {name: _load(name) for name in SCHEMA_FILES}
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    root = schemas["mlarray-1.0.json"]
    schema = root if ref == "mlarray-1.0.json" else {"$ref": root["$id"] + ref}
    return jsonschema.Draft202012Validator(schema, registry=registry)


def validate_core(data):
    _validator().validate(data)


def validate_source_key(data):
    _validator("#/$defs/source_key").validate(data)


def validate_user_key(data):
    _validator("#/$defs/user_key").validate(data)


class TestSchemas(unittest.TestCase):
    def test_schemas_are_valid(self):
        for name in SCHEMA_FILES:
            with self.subTest(schema=name):
                jsonschema.Draft202012Validator.check_schema(_load(name))

    def test_spec_example_validates(self):
        meta = _meta()
        meta.add_history("converted", tool="medvol 2.0.0", settings={"nifti_form": "nibabel"}, warnings=["w"])
        data = meta.to_dict()
        data["writer"] = "mlarray test"
        data["storage"] = {"chunk_size": [1, 20, 30], "block_size": [1, 10, 10], "patch_size": [8, 16, 16],
                           "compression": {"codec": "LZ4HC", "level": 5, "filters": ["SHUFFLE"]}}
        validate_core(data)
        validate_source_key({"source": [entry.to_dict() for entry in meta.source]})
        validate_user_key({"user": meta.user})

    def test_schema_rejects_what_the_spec_forbids(self):
        data = _meta().to_dict()
        data["writer"] = "mlarray test"
        bad = [
            ("axes", [{"name": "z", "type": "space", "step": 1}]),
            ("axes", [{"name": "z", "type": "row"}]),
            ("values", {"kind": "measurement", "labels": {"1": {"name": "x"}}}),
            ("values", {"kind": "label", "labels": {"one": {"name": "x"}}}),
            ("storage", {"compression": {"codec": "LZ4HC", "nthreads": 4}}),
            ("storage", {"chunk_size": [1, 2], "typesize": 4}),
            ("version", "2.0"),
        ]
        for key, value in bad:
            with self.subTest(key=key, value=value):
                broken = dict(data, **{key: value})
                with self.assertRaises(jsonschema.ValidationError):
                    validate_core(broken)
        space = data["spaces"][0]
        both_forms = dict(space, from_array=dict(space["from_array"], scale=[1, 1, 1]))
        with self.assertRaises(jsonschema.ValidationError):
            validate_core(dict(data, spaces=[both_forms]))
        with self.assertRaises(jsonschema.ValidationError):
            validate_core(dict(data, profiles={"ml": {"boxes": {"frame": "index"}}}))
        with self.assertRaises(jsonschema.ValidationError):
            validate_core(dict(data, profiles={"medical": {"dicom": {"Repetition Time": 1}}}))


if __name__ == "__main__":
    unittest.main()
