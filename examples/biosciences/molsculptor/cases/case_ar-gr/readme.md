# <div align="center"><strong>Molsculptor-Agent</strong></div>
### <div align="center">分子生成智能体样例</div>

Molsculptor-Agent是一个依托Molsculptor工程的分子生成智能体，旨在智能化Molsculptor生成过程。

Molsculptor-Agent的主要模块拆解于Molsculptor Pipline，主要如下：
*  pre_infer : 分子预处理模块，将输入的分子通过退火算法优化到可优化空间；
*  decoder_function : 解码模块，利用扩散原理，通过diffusion模型，将编码后的特征进行解码；
*  reward_f : 生成分子打分模块，利用DSDP等工具，对生成的分子和蛋白质结合情况进行评判；
*  selecet_f : 利用NSGA_II算法，根据reward_f分数进行优质分子选择；
*  vae_encoder : 编码模块，对上一轮生成的特征进行下一轮编码；
*  af_encoder_funciton : 二次编码模块；
*  save_results : 结果保存模块，保存生成的分子信息；

Molsculptor-Agent环境部署需求：
*  cd onescience-main
*  pip install -r requirements.txt
*  source ~/packages/dtk-25.04.1/env.sh
*  source ~/packages/dtk-25.04.1/cuda/env.sh
*  pip install . -c constraints.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
*  sh onescience-main/examples/molsculptor/cases/case_ar-gr/mol_pipline.sh

项目在天津计算平台上，完成测试验证。