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

"""AlphaGenome - 统一DNA序列基因组模型，已集成至OneScience。

AlphaGenome 是 Google DeepMind 发布的统一DNA序列模型，能够分析最长100万碱基对的
DNA序列，以单碱基对分辨率预测基因表达、剪接模式、染色质特征和接触图谱。

模块结构:
    model/         - 核心模型组件 (AlphaGenome, SequenceEncoder, 各类Head)
    io/            - 数据输入输出 (BundleName, Dataset, Fasta)
    finetuning/    - 微调工具 (训练步骤、数据管道)
    evals/         - 评估指标 (track预测、回归指标)
    _sdk/          - AlphaGenome SDK (vendored alphagenome v0.6.1)
                     包含数据类型、gRPC客户端、变异/区间评分、可视化和ISM解释工具

使用方式:
    from onescience.flax_models.alphagenome.model import model as alphagenome_model
    from onescience.flax_models.alphagenome.io import dataset
    from onescience.flax_models.alphagenome.finetuning import finetune

    # SDK 数据类型 (原 from alphagenome.data import genome):
    from onescience.flax_models.alphagenome._sdk.data import genome
    from onescience.flax_models.alphagenome._sdk.models import dna_model
    from onescience.flax_models.alphagenome._sdk.models import dna_output
"""

__version__ = '0.1.0'
