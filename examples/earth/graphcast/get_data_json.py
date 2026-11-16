import os
from onescience.utils.YParams import YParams
import json
import sys


def main():
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg_data = YParams(config_file_path, "datapipe")
    metadata = {
        "coords": {
            "channel": {str(i): name for i, name in enumerate(cfg_data.dataset.channels)}
        }
    }
    with open('./data.json', "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved.")



if __name__ == '__main__':
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
