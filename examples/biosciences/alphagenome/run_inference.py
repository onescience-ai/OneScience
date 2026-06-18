# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AlphaGenome 推理示例脚本。

该脚本演示如何使用 AlphaGenome 对基因组区间进行多模态预测，包括:
  - 基因表达 (RNA-seq, CAGE)
  - 染色质可及性 (ATAC-seq, DNase-seq)
  - 转录因子结合 (ChIP-seq)
  - Hi-C 接触图谱
  - 剪接位点

用法:
    # 使用 Kaggle Hub 自动下载模型（默认模式）
    python run_inference.py

    # 指定本地参考基因组
    python run_inference.py \\
        --fasta_path /path/to/GRCh38.fa \\
        --chromosome chr19 \\
        --start 10587331 \\
        --end 11635907 \\
        --output_dir ./outputs
"""

import os
import pathlib

from absl import app
from absl import flags
from absl import logging

# from alphagenome.data import genome
from onescience.flax_models.alphagenome._sdk.data import genome
from onescience.flax_models.alphagenome.model import dna_model as dna_model_types
import numpy as np

# OneScience 命名空间导入
from onescience.flax_models.alphagenome.model.dna_model import (
    AlphaGenomeModel,
    create,
    create_from_kaggle,
    OrganismSettings,
)
from onescience.flax_models.alphagenome.model.one_hot_encoder import DNAOneHotEncoder

FLAGS = flags.FLAGS

flags.DEFINE_string('fasta_path', None, '参考基因组 FASTA 文件路径 (需要 .fai 索引)。')
flags.DEFINE_string('model_dir', None, 'AlphaGenome 模型权重目录。若未指定则从 Kaggle Hub 下载。')
flags.DEFINE_string('chromosome', 'chr1', '染色体名称，例如 chr1。')
flags.DEFINE_integer('start', 1_000_000, '区间起始位置 (0-based)。')
flags.DEFINE_integer('end', 2_048_576, '区间终止位置 (exclusive)。序列长度需为 1048576 的整数倍。')
flags.DEFINE_string('output_dir', './outputs', '输出目录。')
flags.DEFINE_enum(
    'organism', 'HOMO_SAPIENS',
    ['HOMO_SAPIENS', 'MUS_MUSCULUS'],
    '目标生物体。'
)
flags.DEFINE_enum(
    'model_version', 'FOLD_0',
    ['FOLD_0', 'FOLD_1', 'FOLD_2', 'FOLD_3', 'FOLD_4', 'all_folds'],
    '模型版本（交叉验证折叠或 all_folds 集成）。'
)


def main(_):
    output_dir = pathlib.Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    organism = dna_model_types.Organism[FLAGS.organism]
    model_version = FLAGS.model_version
    interval = genome.Interval(
        chromosome=FLAGS.chromosome,
        start=FLAGS.start,
        end=FLAGS.end,
    )

    # 加载模型
    logging.info('加载模型权重...')
    if FLAGS.model_dir:
        # 从本地检查点加载
        organism_settings = None
        if FLAGS.fasta_path:
            organism_settings = {
                organism: OrganismSettings(
                    fasta_path=FLAGS.fasta_path,
                ),
            }
        alphagenome_model = create(
            checkpoint_path=FLAGS.model_dir,
            organism_settings=organism_settings,
        )
    else:
        # 从 Kaggle Hub 下载
        alphagenome_model = create_from_kaggle(model_version)

    # 运行推理
    logging.info('运行推理: %s', interval)
    if FLAGS.fasta_path or not FLAGS.model_dir:
        # 模型已配置 FASTA，可以用 predict_interval
        predictions = alphagenome_model.predict_interval(
            interval,
            organism=organism,
            requested_outputs={
                dna_model_types.OutputType.ATAC,
                dna_model_types.OutputType.DNASE,
                dna_model_types.OutputType.CAGE,
                dna_model_types.OutputType.RNA_SEQ,
                dna_model_types.OutputType.CHIP_TF,
                dna_model_types.OutputType.CHIP_HISTONE,
            },
            ontology_terms=None,
        )
    else:
        # 演示模式：生成随机序列，使用 predict_sequence
        logging.warning('未提供 fasta_path，使用随机序列进行演示。')
        rng = np.random.default_rng(42)
        bases = np.array(['A', 'C', 'G', 'T'])
        seq_len = interval.end - interval.start
        dna_sequence = ''.join(rng.choice(bases, size=seq_len))
        predictions = alphagenome_model.predict_sequence(
            dna_sequence,
            organism=organism,
            requested_outputs={
                dna_model_types.OutputType.ATAC,
                dna_model_types.OutputType.DNASE,
                dna_model_types.OutputType.RNA_SEQ,
            },
            ontology_terms=None,
            interval=interval,
        )

    # 保存预测结果
    logging.info('保存预测结果至 %s', output_dir)
    for attr_name in ['atac', 'dnase', 'cage', 'rna_seq', 'chip_tf',
                       'chip_histone', 'contact_maps', 'procap']:
        output = getattr(predictions, attr_name, None)
        if output is not None:
            output_path = output_dir / f'{attr_name}.npy'
            np.save(output_path, np.array(output.values))
            logging.info('  保存 %s: shape=%s', attr_name,
                         np.array(output.values).shape)

    logging.info('推理完成。')


if __name__ == '__main__':
    app.run(main)
