from fastapi import FastAPI
from pydantic import BaseModel
import sys
import random
import json
from server_alphafold import main,_OUTPUT_DIR
from absl import flags
from save_result import save_structure_results
import os

app_fastapi = FastAPI()
user_id_dict = {}

# 输入参数模型
class InputData_infer(BaseModel):
    user_id : str
    json_dict: dict

class InputData_show(BaseModel):
    user_id : str
    output_file: str

#命令行参数模拟
sys.argv = [
        "run_alphaflod.py",
        '--json_path=./af_input/7r6r_data.json',
        '--model_dir=/work/home/onescience2025/panpy/af3/',
        '--output_dir=./af_output',
        '--run_data_pipeline=false',
        '--flash_attention_implementation=xla'
    ]
flags.FLAGS(sys.argv)
#由于命令行参数不能修改，需要在主函数重新组织参数逻辑
#absl.app的运行方式也需要修改成普通函数调用，其他重要参数作为普通参数传入

def write_json(indic):
    file_root = "./input_files_tmp"
    file_name = "input_" + str(random.randint(0,1000000))+ ".json"
    file_root_ = os.path.join(file_root, file_name)
    while os.path.exists(file_root_):
        file_name = "input_" + str(random.randint(0,1000000))+ ".json"
        file_root_ = os.path.join(file_root, file_name)
    with open(file_root_, "w") as f:
        json.dump(indic, f, ensure_ascii=False)
    return file_root_

@app_fastapi.post("/af3")
def af3_infer(data: InputData_infer):
    try:
        flags.mark_flags_as_required(['output_dir'])
        json_path = write_json(data.json_dict)
        main(None, json_path)
        user_id_dict[data.user_id] = [_OUTPUT_DIR.value, data.json_dict['name']]
        return {"state":"done" , "result": f"af3 results in {_OUTPUT_DIR.value}"}
    except Exception as e:
        print(e)
        return {"state":"error", "result":"some error occured"}

@app_fastapi.post("/af3_save_result")
def af3_save_result(data:InputData_show):
    inference_ouput = os.path.join(user_id_dict[data.user_id][0],user_id_dict[data.user_id][1])
    output_file = data.output_file
    print("inference_output:",inference_ouput)
    print("output_file:",output_file)
    save_structure_results(inference_ouput,output_file)
    return {"state":"done", "result":output_file}
