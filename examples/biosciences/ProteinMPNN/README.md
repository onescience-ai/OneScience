# ProteinMPNN

![ProteinMPNN](https://docs.google.com/drawings/d/e/2PACX-1vTtnMBDOq8TpHIctUfGN8Vl32x5ISNcPKlxjcQJF2q70PlaH2uFlj2Ac4s3khnZqG1YxppdMr0iTyk-/pub?w=889&h=358)

阅读 [ProteinMPNN 论文](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1)。

### 全蛋白骨架模型：
- `vanilla_model_weights/v_48_002.pt, v_48_010.pt, v_48_020.pt, v_48_030.pt`
- `soluble_model_weights/v_48_010.pt, v_48_020.pt`

### 仅 CA 模型：
- `ca_model_weights/v_48_002.pt, v_48_010.pt, v_48_020.pt`
- 启用 `--ca_only` 参数以使用这些模型。

### 辅助脚本：
- `helper_scripts`：用于解析 PDB 文件的辅助函数，指定需要设计的链、固定的残基、添加氨基酸偏差、绑定残基等。

### 代码结构：
* `protein_mpnn_run.py` - 初始化并运行模型的主脚本。
* `protein_mpnn_utils.py` - 主脚本的辅助函数。
* `examples/` - 简单的代码示例。
* `inputs/` - 示例的输入 PDB 文件。
* `outputs/` - 示例的输出。
* `colab_notebooks/` - Google Colab 示例。
* `training/` - 重训模型的代码和数据。

---

### `protein_mpnn_run.py` 输入参数：
```bash
argparser.add_argument("--suppress_print", type=int, default=0, help="0 为 False，1 为 True")
argparser.add_argument("--ca_only", action="store_true", default=False, help="解析仅 CA 的结构并使用仅 CA 的模型（默认：false）")
argparser.add_argument("--path_to_model_weights", type=str, default="", help="模型权重文件夹的路径；")
argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN 模型名称：v_48_002, v_48_010, v_48_020, v_48_030；v_48_010=具有 48 条边且噪声为 0.10A 的版本")
argparser.add_argument("--use_soluble_model", action="store_true", default=False, help="加载仅在可溶蛋白上训练的 ProteinMPNN 权重。")
argparser.add_argument("--seed", type=int, default=0, help="如果设置为 0，则随机选择种子；")
argparser.add_argument("--save_score", type=int, default=0, help="0 为 False，1 为 True；保存分数 = -log_prob 到 npy 文件")
argparser.add_argument("--path_to_fasta", type=str, default="", help="以 fasta 格式提供的输入序列的路径；例如：GGGGGG/PPPPS/WWW，链 A、B、C 按字母顺序排序并用 / 分隔")
argparser.add_argument("--save_probs", type=int, default=0, help="0 为 False，1 为 True；保存 MPNN 预测的每个位点概率")
argparser.add_argument("--score_only", type=int, default=0, help="0 为 False，1 为 True；仅对输入的骨架-序列对进行打分")
argparser.add_argument("--conditional_probs_only", type=int, default=0, help="0 为 False，1 为 True；仅输出条件概率 p(s_i 给定其余序列和骨架)")
argparser.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 为 False，1 为 True；如果为真，则仅输出条件概率 p(s_i 给定骨架)")
argparser.add_argument("--unconditional_probs_only", type=int, default=0, help="0 为 False，1 为 True；一次前向传递输出无条件概率 p(s_i 给定骨架)")
argparser.add_argument("--backbone_noise", type=float, default=0.00, help="骨架原子的高斯噪声标准差")
argparser.add_argument("--num_seq_per_target", type=int, default=1, help="每个目标生成的序列数")
argparser.add_argument("--batch_size", type=int, default=1, help="批量大小；如果使用 Titan 或 Quadro GPU，可以设置更高的值；如果显存不足，则需要减小批量大小")
argparser.add_argument("--max_length", type=int, default=200000, help="最大序列长度")
argparser.add_argument("--sampling_temp", type=str, default="0.1", help="温度字符串，例如：0.2 0.25 0.5。氨基酸的采样温度。建议值为 0.1、0.15、0.2、0.25、0.3。较高的温度将导致更多样化的序列。")
argparser.add_argument("--out_folder", type=str, help="输出序列的文件夹路径，例如：/home/out/")
argparser.add_argument("--pdb_path", type=str, default='', help="需要设计的单个 PDB 文件的路径")
argparser.add_argument("--pdb_path_chains", type=str, default='', help="定义需要设计的单个 PDB 文件中的链")
argparser.add_argument("--jsonl_path", type=str, help="解析的 pdb 文件所在文件夹路径")
argparser.add_argument("--chain_id_jsonl", type=str, default='', help="指定哪些链需要设计，哪些链需要固定的字典路径，如果未指定，则所有链将被设计。")
argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="包含固定位置的字典路径")
argparser.add_argument("--omit_AAs", type=list, default='X', help="指定哪些氨基酸在生成的序列中应被省略，例如：'AC' 将省略丙氨酸和半胱氨酸。")
argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="包含氨基酸组成偏差的字典路径，例如：{A: -1.1, F: 0.7} 会使 A 的概率降低，而 F 的概率增加。")
argparser.add_argument("--bias_by_res_jsonl", default='', help="包含每个位置偏差的字典路径。")
argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="指定需要从设计中省略的特定链位置的氨基酸字典路径")
argparser.add_argument("--pssm_jsonl", type=str, default='', help="包含 pssm 的字典路径")
argparser.add_argument("--pssm_multi", type=float, default=0.0, help="一个介于 [0.0, 1.0] 之间的值，0.0 表示不使用 pssm，1.0 忽略 MPNN 预测")
argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="一个介于 -inf 和 +inf 之间的值，用于限制每个位置的氨基酸")
argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 为 False，1 为 True")
argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 为 False，1 为 True")
argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="包含绑定位点的字典路径")
```

-----------------------------------------------------------------------------------------------------
提供的 `examples/`:
* `submit_example_1.sh` - 简单的单体示例
* `submit_example_2.sh` - 简单的多链示例
* `submit_example_3.sh` - 直接从 .pdb 路径运行
* `submit_example_3_score_only.sh` -  仅返回分数（模型的不确定性）
* `submit_example_3_score_only_from_fasta.sh` - 仅返回分数（模型的不确定性），从 fasta 文件加载序列
* `submit_example_4.sh` - 固定一些残基位置
* `submit_example_4_non_fixed.sh` - 指定需要设计的残基位置
* `submit_example_5.sh` - 将一些位置绑在一起（对称性问题）
* `submit_example_6.sh` - 寡聚体示例
* `submit_example_7.sh` - 返回序列的无条件概率（类似 PSSM）
* `submit_example_8.sh` - 添加氨基酸偏差
* `submit_example_pssm.sh` - 在设计序列时使用 PSSM 偏差


-----------------------------------------------------------------------------------------------------
输出示例:
```
>3HTN, score=1.1705, global_score=1.2045, fixed_chains=['B'], designed_chains=['A', 'C'], model_name=v_48_020, git_hash=015ff820b9b5741ead6ba6795258f35a9c15e94b, seed=37
NMYSYKKIGNKYIVSINNHTEIVKALNAFCKEKGILSGSINGIGAIGELTLRFFNPKTKAYDDKTFREQMEISNLTGNISSMNEQVYLHLHITVGRSDYSALAGHLLSAIQNGAGEFVVEDYSERISRTYNPDLGLNIYDFER/NMYSYKKIGNKYIVSINNHTEIVKALNAFCKEKGILSGSINGIGAIGELTLRFFNPKTKAYDDKTFREQMEISNLTGNISSMNEQVYLHLHITVGRSDYSALAGHLLSAIQNGAGEFVVEDYSERISRTYNPDLGLNIYDFER
>T=0.1, sample=1, score=0.7291, global_score=0.9330, seq_recovery=0.5736
NMYSYKKIGNKYIVSINNHTEIVKALKKFCEEKNIKSGSVNGIGSIGSVTLKFYNLETKEEELKTFNANFEISNLTGFISMHDNKVFLDLHITIGDENFSALAGHLVSAVVNGTCELIVEDFNELVSTKYNEELGLWLLDFEK/NMYSYKKIGNKYIVSINNHTDIVTAIKKFCEDKKIKSGTINGIGQVKEVTLEFRNFETGEKEEKTFKKQFTISNLTGFISTKDGKVFLDLHITFGDENFSALAGHLISAIVDGKCELIIEDYNEEINVKYNEELGLYLLDFNK
>T=0.1, sample=2, score=0.7414, global_score=0.9355, seq_recovery=0.6075
NMYKYKKIGNKYIVSINNHTEIVKAIKEFCKEKNIKSGTINGIGQVGKVTLRFYNPETKEYTEKTFNDNFEISNLTGFISTYKNEVFLHLHITFGKSDFSALAGHLLSAIVNGICELIVEDFKENLSMKYDEKTGLYLLDFEK/NMYKYKKIGNKYVVSINNHTEIVEALKAFCEDKKIKSGTVNGIGQVSKVTLKFFNIETKESKEKTFNKNFEISNLTGFISEINGEVFLHLHITIGDENFSALAGHLLSAVVNGEAILIVEDYKEKVNRKYNEELGLNLLDFNL
```
* `score` - 设计的残基的平均负对数概率
* `global score` - 所有链中所有残基的平均负对数概率
* `fixed_chains` - 未设计的链（固定）
* `designed_chains` - 重新设计的链
* `model_name/CA_model_name` - 用于生成结果的模型名称，例如:`v_48_020`
* `git_hash` - 生成输出时使用的 GitHub 版本
* `seed` - 随机种子
* `T=0.1` - 使用温度 0.1 采样序列
* `sample` - 序列采样编号 1、2、3...等

-----------------------------------------------------------------------------------------------------
```
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},  
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```
-----------------------------------------------------------------------------------------------------
