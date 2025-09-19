exp1 = """
算法: inference_pipeline (分子生成与评估推理流程)

输入: config - 配置文件路径

开始
    // 步骤 1: 初始化
    调用 init_f 函数，使用配置 config 进行全局初始化
    调用 pre_infer 函数，准备推理所需的初始分子和状态
    
    // 步骤 2: 执行多轮搜索迭代
    设定搜索总步数为 args.search_config.search_steps
    对于 每一轮迭代 step_it 从 0 到 (搜索总步数 - 1):
        调用 decoder_fc 函数: 将当前编码的分子解码为可读形式（如 SMILES），并更新缓存
        调用 reward_f 函数: 对解码后的分子进行评估，计算其奖励分数（如活性、类药性等）
        调用 select_f 函数: 根据评估分数和预设的约束条件（如分子量、毒性），选择表现最优的分子
        调用 vae_encoder 函数，传入当前迭代步数 step_it: 将选中的分子重新编码，为下一轮迭代做准备
    
    // 步骤 3: 最终处理
    调用 af_encoder_function 函数: 对最终选定的分子集进行最终编码（可能使用不同的编码器）
    
    // 步骤 4: 保存结果
    调用 save_results 函数: 将最终的分子、分数、日志等结果保存到磁盘
    
结束

// 主程序入口
调用 inference_pipeline 函数，传入配置文件路径 "../../application/molsculptor/config.ini"
"""
examples = [exp1]
