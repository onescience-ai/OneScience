"""Packaging configuration for TargetDiff utility data."""

TARGETDIFF_PACKAGE_DATA = {
    "onescience.utils.targetdiff.evaluation": [
        "fpscores.pkl.gz",
    ],
}

TARGETDIFF_MANIFEST_RULES = [
    "include src/onescience/utils/targetdiff/evaluation/fpscores.pkl.gz",
]


def get_package_data():
    return TARGETDIFF_PACKAGE_DATA


def get_manifest_rules():
    return TARGETDIFF_MANIFEST_RULES

