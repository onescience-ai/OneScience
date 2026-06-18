#!/usr/bin/env python3
"""
AlphaGenome package config.

Expose runtime metadata files through the normal packaging pipeline so
install scripts do not need to copy files into site-packages manually.
"""

ALPHAGENOME_PACKAGE_DATA = {
    "onescience.flax_models.alphagenome.model.metadata": [
        "*.textproto",
    ],
}

ALPHAGENOME_MANIFEST_RULES = [
    "recursive-include src/onescience/flax_models/alphagenome/model/metadata *.textproto",
]


def get_package_data():
    return ALPHAGENOME_PACKAGE_DATA


def get_manifest_rules():
    return ALPHAGENOME_MANIFEST_RULES
