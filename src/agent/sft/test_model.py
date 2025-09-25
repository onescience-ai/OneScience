from transformers import pipeline

model_path_base = (
    "/work/home/onescience2025/.cache/modelscope/hub/Qwen/Qwen2.5-7B-Instruct"
)
model_path_ft500 = "/work/home/onescience2025/deepseek/open-r1/data/Qwen2.5-7B-Open-R1-Distill-lora-chinese-SFT/checkpoint-500"
model_path_ft2604 = "/work/home/onescience2025/deepseek/open-r1/data/Qwen2.5-7B-Open-R1-Distill-lora-chinese-SFT/checkpoint-2604"
model_path_ft2000_med = "/work/home/onescience2025/deepseek/open-r1/data/Qwen2.5-7B-Open-R1-Distill-lora-chinese-Med/checkpoint-2000"
choose_model = model_path_base

# question = "在大学里应该谈恋爱吗？"
# question = '针对一名28岁女性患者，她左小腿挫伤12小时，伤口有分泌物，骨折端外露，小腿成角畸形，描述她的最佳处理方法。'
# question = '一位中年男性患者因胸骨后剧烈疼痛40分钟来急诊，心电图显示急性广泛前壁心肌梗死，血压80/45mmHg，正在静脉滴注多巴胺及硝普钠。为了明确左心功能状态，应该进行哪项检查？'
# question = '在单卵双生中，一人40岁以前出现糖尿病的情况下，另一人多数会发生哪种类型的糖尿病？'
question = """Fortran 编译器：
➢ 新增 OpenACC serial、atomic、declare、routine 指令支持。
➢ 提升 14 个设备端数学函数的计算精度。
➢ OpenMP、OpenACC 支持 hipprof 工具打印性能日志。新增-Minfo 编译选
项，用于打印编译日志。添加-g 选项，可以打印出异构参数名称、文件
行数信息。
根据以上内容，回答dtk25.04中关于Fortran编译器有什么新功能"""

messages = [
    {"role": "user", "content": question},
]
pipe = pipeline(
    "text-generation", model=choose_model, trust_remote_code=True, device="cuda"
)
out = pipe(
    messages,
    max_new_tokens=500,  # 设置最大生成token数
    do_sample=True,  # 启用采样(可选)
    temperature=0.7,  # 控制随机性(可选)
    top_p=0.9,  # 核采样参数(可选)
)

# print(out)
print("q:", out[0]["generated_text"][0]["content"])
print("*****推理模型:", choose_model)
print("a:", out[0]["generated_text"][1]["content"])
