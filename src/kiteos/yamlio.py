# Copyright 2026 Kite ML.
"""YAML loading that refuses duplicate keys (PyYAML silently keeps the last one).

A duplicated ``zones:`` in a fence file (for example an uncommented example block above
``zones: []``), or a second ``fence:`` in a station file, would otherwise drop or replace a
safety-relevant setting without any error. Depends on PyYAML only.
"""

from __future__ import annotations

from typing import Any

import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that raises ``yaml.constructor.ConstructorError`` on duplicate keys."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    loader.flatten_mapping(node)
    seen: dict[Any, int] = {}
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
        try:
            hash(key)
        except TypeError:
            continue  # the schema reports unhashable keys
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r} (first on line {seen[key] + 1})",
                key_node.start_mark,
            )
        seen[key] = key_node.start_mark.line
    return loader.construct_mapping(node, deep=deep)


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_yaml_unique(text: str) -> Any:
    """Parse YAML like ``yaml.safe_load`` but raise ``yaml.YAMLError`` on duplicate keys."""
    return yaml.load(text, Loader=UniqueKeyLoader)  # noqa: S506 - a SafeLoader subclass
