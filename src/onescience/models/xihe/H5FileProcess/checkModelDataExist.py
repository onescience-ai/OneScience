import json
from onescience.utils.YParams import YParams


CONFIG_PATH = '/public/home/onescience2025404/zhaozhn/onescience/examples/earth/'


def check_exsit(model):
    cfg = YParams(f'{CONFIG_PATH}/{model}/conf/config.yaml', "datapipe")
    metadata_path = "/public/onestore/onedatasets/ERA5/newh5/metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    variables = metadata['variables']

    channels = cfg.dataset.channels
    # 检查 channels 是否都在 metadata.variables 中
    missing = [ch for ch in channels if ch not in variables]
    if missing:
        print(f"❌ {model}: Missing {len(missing)} required variables in metadata: {missing}\n\n")
    else:
        print(f"✅ {model} needs {len(channels)} variables are all existing...\n\n")

if __name__ == "__main__":
    check_exsit('fourcastnet')
    check_exsit('fengwu')
    check_exsit('fuxi')
    check_exsit('graphcast')
    check_exsit('pangu_weather')