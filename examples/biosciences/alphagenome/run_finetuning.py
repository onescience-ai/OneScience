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

"""AlphaGenome 微调示例脚本。

该脚本演示如何在自定义基因组数据上微调 AlphaGenome 模型，
适用于以下场景:
  - 新细胞类型/组织的ATAC-seq/ChIP-seq信号预测
  - 特定实验数据的适配
  - 迁移学习到新物种

数据要求:
  - 参考基因组 FASTA 文件
  - BigWig 格式的信号轨迹文件
  - 训练区间 CSV 文件 (列: chromosome, start, end)

用法:
    python run_finetuning.py \\
        --fasta_path /path/to/GRCh38.fa \\
        --regions_csv /path/to/regions.csv \\
        --bigwig_paths /path/to/atac.bw,/path/to/atac2.bw \\
        --output_dir ./finetuned_model \\
        --num_steps 1000 \\
        --batch_size 2
"""

import pathlib

from absl import app
from absl import flags
from absl import logging

from onescience.flax_models.alphagenome._sdk.data import fold_intervals
from onescience.flax_models.alphagenome._sdk.models import dna_model as dna_model_types
import jax
import optax
import orbax.checkpoint as ocp

# OneScience 命名空间导入
from onescience.flax_models.alphagenome.finetuning.finetune import (
    get_dataset_iterator,
    get_train_step,
)
from onescience.flax_models.alphagenome.evals.track_prediction import load_model
from onescience.flax_models.alphagenome.model.metadata.metadata import load as load_metadata

FLAGS = flags.FLAGS

flags.DEFINE_string('fasta_path', None, '参考基因组 FASTA 文件路径。', required=True)
flags.DEFINE_string('regions_csv', None, '训练区间 CSV 文件路径 (列: chromosome,start,end)。', required=True)
flags.DEFINE_list('bigwig_paths', None, 'BigWig 信号文件路径列表，逗号分隔。', required=True)
flags.DEFINE_string('model_dir', None, '预训练模型权重目录。若未指定则从 Kaggle Hub 下载。')
flags.DEFINE_string('output_dir', './finetuned_model', '微调后模型的保存目录。')
flags.DEFINE_integer('num_steps', 1000, '训练步数。')
flags.DEFINE_integer('batch_size', 2, '训练批次大小。')
flags.DEFINE_float('learning_rate', 1e-5, '初始学习率。')
flags.DEFINE_integer('log_every', 50, '每隔多少步打印一次训练日志。')
flags.DEFINE_integer('save_every', 200, '每隔多少步保存一次检查点。')
flags.DEFINE_enum(
    'model_version', 'FOLD_0',
    ['FOLD_0', 'FOLD_1', 'FOLD_2', 'FOLD_3', 'FOLD_4'],
    '预训练模型版本。'
)
flags.DEFINE_enum(
    'organism', 'HOMO_SAPIENS',
    ['HOMO_SAPIENS', 'MUS_MUSCULUS'],
    '目标生物体。'
)


def main(_):
    output_dir = pathlib.Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_version = dna_model_types.ModelVersion[FLAGS.model_version]
    organism = dna_model_types.Organism[FLAGS.organism]

    logging.info('JAX 设备: %s', jax.devices())
    logging.info('微调配置: lr=%.2e, steps=%d, batch_size=%d',
                 FLAGS.learning_rate, FLAGS.num_steps, FLAGS.batch_size)

    # 加载预训练模型
    logging.info('加载预训练模型...')
    params, state, predict_fn = load_model(model_version)

    # 加载输出元数据
    output_metadata = load_metadata(organism)

    # 构建优化器 (带 warmup 的余弦退火学习率)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.learning_rate,
        warmup_steps=100,
        decay_steps=FLAGS.num_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    )
    opt_state = optimizer.init(params)

    # 构建微调训练步骤
    train_step = get_train_step(
        predict_fn=predict_fn,
        optimizer=optimizer,
    )

    # 构建数据集迭代器
    logging.info('构建微调数据集迭代器...')
    dataset_iter = get_dataset_iterator(
        batch_size=FLAGS.batch_size,
        sequence_length=1_048_576,
        output_metadata=output_metadata,
        model_version=model_version,
        subset=fold_intervals.Subset.TRAIN,
        organism=organism,
        fasta_path=FLAGS.fasta_path,
        example_regions_path=FLAGS.regions_csv,
    )

    # 检查点管理器
    checkpointer = ocp.CheckpointManager(
        output_dir / 'checkpoints',
        options=ocp.CheckpointManagerOptions(max_to_keep=3),
    )

    # 训练循环
    logging.info('开始微调训练...')
    for step, batch in enumerate(dataset_iter):
        if step >= FLAGS.num_steps:
            break

        params, state, opt_state, metrics = train_step(
            params, state, opt_state, batch
        )

        if step % FLAGS.log_every == 0:
            loss = float(metrics.get('loss', float('nan')))
            logging.info('Step %d/%d | loss=%.4f', step, FLAGS.num_steps, loss)

        if step % FLAGS.save_every == 0 and step > 0:
            checkpointer.save(step, args=ocp.args.StandardSave({'params': params, 'state': state}))
            logging.info('检查点已保存 (step=%d)', step)

    # 保存最终模型
    checkpointer.save(
        FLAGS.num_steps,
        args=ocp.args.StandardSave({'params': params, 'state': state}),
    )
    logging.info('微调完成，最终模型已保存至: %s', output_dir / 'checkpoints')


if __name__ == '__main__':
    app.run(main)
