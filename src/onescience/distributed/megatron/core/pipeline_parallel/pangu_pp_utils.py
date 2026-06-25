# pangu_pp_utils.py
from onescience.distributed.megatron.core.utils import get_attr_wrapped_model

def is_pangu_model(model):
    try:
        meta = get_attr_wrapped_model(model, "meta")
        if hasattr(meta, "name"):
            return meta.name.lower() == "pangu"
        return False
    except:
        return False
