"""JSON处理模块

提供统一的JSON文件解析、写入和转换功能，
支持蛋白质结构预测模型（Protenix/AlphaFold3等）的输入格式。

使用示例
--------
>>> from onescience.datapipes.biology.common.json import (
...     JSONParser, JSONWriter, JSONConverter,
...     ProteinJSONParser, ProteinJSONWriter, ProteinJSONConverter,
...     JSONData, JSONWriteConfig,
...     ProtenixJSONData, ProtenixJSONFeaturizer
... )

>>> # 解析JSON文件
>>> json_data = JSONParser.parse_file("input.json")
>>> print(json_data.name)
>>> print(json_data.get_sequences())

>>> # 创建JSON数据
>>> writer = ProteinJSONWriter()
>>> sequences = [
...     writer.create_protein_entry("MKTAYIAKQRQISFVKSHFSRQ", count=1),
...     writer.create_ligand_entry("CCD_ATP", count=1)
... ]
>>> writer.write_structure(sequences, "output.json", name="sample1")

>>> # 转换格式
>>> converter = ProteinJSONConverter()
>>> normalized = converter.normalize(json_data)
>>> composition = converter.calculate_composition(json_data)

>>> # 特征提取（需要Protenix依赖）
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
    # 数据类
    "JSONData",
    "JSONWriteConfig",
    # 解析器
    "JSONParser",
    "ProteinJSONParser",
    # 写入器
    "JSONWriter",
    "ProteinJSONWriter",
    # 转换器
    "JSONConverter",
    "ProteinJSONConverter",
    "JSONBatchProcessor",
]

# # 如果Protenix可用，添加到__all__
# if _PROTEINIX_AVAILABLE:
#     __all__.extend([
#         # Protenix特征提取
#         "ProtenixJSONData",
#         "ProtenixJSONFeaturizer",
#     ])
