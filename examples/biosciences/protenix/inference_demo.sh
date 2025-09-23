N_sample=5
N_step=200
N_cycle=10
seed=101
use_deepspeed_evo_attention=false
input_json_path="./infer_datasets/example.json"
# wget -P /af3-dev/release_model/ https://af3-dev.tos-cn-beijing.volces.com/release_model/model_v0.5.0.pt
load_checkpoint_path="/public/onestore/onemodels/Protenix/model_v0.5.0.pt"
dump_dir="./output"

export PYTHONPATH=$(pwd):$PYTHONPATH
export DATA_ROOT_DIR=/public/onestore/onedatasets/protenix/
export HIP_VISIBLE_DEVICES=0 # 指定运行gpu
python3 runner/inference.py \
--seeds ${seed} \
--dtype bf16 \
--num_workers 8 \
--load_checkpoint_path ${load_checkpoint_path} \
--dump_dir ${dump_dir} \
--input_json_path ${input_json_path} \
--model.N_cycle ${N_cycle} \
--sample_diffusion.N_sample ${N_sample} \
--sample_diffusion.N_step ${N_step}
