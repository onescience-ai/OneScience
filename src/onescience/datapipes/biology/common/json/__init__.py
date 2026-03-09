"""JSON processing module.

Provides unified JSON file parsing, writing, and conversion functionality,
supporting input formats for protein structure prediction models (Protenix/AlphaFold3, etc.).

Usage Examples
--------
>>> from onescience.datapipes.biology.common.json import (
...     JSONParser, JSONWriter, JSONConverter,
...     ProteinJSONParser, ProteinJSONWriter, ProteinJSONConverter,
...     JSONData, JSONWriteConfig,
...     ProtenixJSONData, ProtenixJSONFeaturizer
... )

>>> # Parse JSON file
>>> json_data = JSONParser.parse_file("input.json")
>>> print(json_data.name)
>>> print(json_data.get_sequences())

>>> # Create JSON data
>>> writer = ProteinJSONWriter()
>>> sequences = [
...     writer.create_protein_entry("MKTAYIAKQRQISFVKSHFSRQ", count=1),
...     writer.create_ligand_entry("CCD_ATP", count=1)
... ]
>>> writer.write_structure(sequences, "output.json", name="sample1")

>>> # Convert format
>>> converter = ProteinJSONConverter()
>>> normalized = converter.normalize(json_data)
>>> composition = converter.calculate_composition(json_data)

>>> # Feature extraction (requires Protenix dependency)
>>> featurizer = ProtenixJSONFeaturizer("input.json")
>>> features, atom_array, token_array = featurizer.get_features()
"""

from onescience.datapipes.biology.common.json.json_parser import (
    JSONData,
    JSONParser,
    ProteinJSONParser,
)

from onescience.datapipes.biology.common.json.json_writer import (
    JSONWriteConfig,
    JSONWriter,
    ProteinJSONWriter,
)

from onescience.datapipes.biology.common.json.json_converter import (
    JSONConverter,
    ProteinJSONConverter,
    JSONBatchProcessor,
)

# try:
#     from onescience.datapipes.biology.common.json.protenix_json_to_feature import (
#         ProtenixJSONData,
#         ProtenixJSONFeaturizer,
#     )
#     _PROTEINIX_AVAILABLE = True
# except ImportError:
#     _PROTEINIX_AVAILABLE = False
#     ProtenixJSONData = None  # type: ignore
#     ProtenixJSONFeaturizer = None  # type: ignore

__all__ = [
    # Data classes
    "JSONData",
    "JSONWriteConfig",
    # Parsers
    "JSONParser",
    "ProteinJSONParser",
    # Writers
    "JSONWriter",
    "ProteinJSONWriter",
    # Converters
    "JSONConverter",
    "ProteinJSONConverter",
    "JSONBatchProcessor",
]

# # Add to __all__ if Protenix is available
# if _PROTEINIX_AVAILABLE:
#     __all__.extend([
#         # Protenix feature extraction
#         "ProtenixJSONData",
#         "ProtenixJSONFeaturizer",
#     ])
