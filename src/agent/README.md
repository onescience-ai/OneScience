<div align="center">

# OneScience Agent

</div>

旨在构建基于LLM的通用Science智能体，辅助Science工作，提升其效率和性能。

## 近期更新

- 2025-09-24 实现OneScience智能体

## 安装使用

### 安装OneScience
本项目 DCU 显卡所需的torch、torchvision、apex、dgl库下载地址： https://developer.hpccube.com/tool/

```
cd onescience
pip install -e .

```
注：特殊依赖库需要与dtk版本对应。

## 快速开始

### Agent使用
#### 本地部署LLM服务
- 节点申请：

    ```salloc -p k100ai --gres=dcu:4 -n 32```
- 登录节点：

    ```ssh a02r3n20```
- 加载环境：

    ```source ~/env_dtk25041.sh```
- 激活运行环境：

    ```conda activate vllm```
- 启动服务：

    ```python -m vllm.entrypoints.openai.api_server --model "/work/home/onescience2025/Qwen/Qwen3-32B" --served-model-name "Qwen3-32B" --enable-auto-tool-choice --tool-call-parser hermes --tensor-parallel-size 4```

登录节点名称由节点申请获取，实际部署中替换为申请的节点名称。启动服务时的模型路径需要替换为实际部署的模型路径。

#### 运行Agent

```
from agent import OnescienceAgent

agent = OnescienceAgent(job_name='molsculptor', llm_server='http://a02r2n09:8000/v1', log_dir='./logs')
agent.run("请使用2个搜索步数帮我设计一个蛋白小分子结构，各参数初始化的配置文件路径为'./application/molsculptor/config.ini")

```

注意：目前只支持Qwen3-32B模型




