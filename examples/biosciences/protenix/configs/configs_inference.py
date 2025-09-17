# pylint: disable=C0114
from onescience.models.protenix.config.extend_types import ListValue, RequiredValue

# The model will be download to the following dir if not exists:
# "./release_data/checkpoint/model_v0.5.0.pt"
inference_configs = {
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": True,
    "input_json_path": RequiredValue(str),
    "load_checkpoint_path": RequiredValue(str),
    "num_workers": 16,
    "use_msa": True,
}
