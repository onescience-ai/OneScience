from . import Transolver
from . import LSM
from . import FNO
from . import U_Net
from . import Transformer
from . import Factformer
from . import Swin_Transformer
from . import Galerkin_Transformer
from . import GNOT
from . import U_NO
from . import U_FNO
from . import F_FNO
from . import ONO
from . import MWT
from . import GraphSAGE
from . import Graph_UNet
from . import PointNet
from . import DeepONet
from . import MeshGraphNet
from . import RegDGCNN
from . import GFNO


def get_model(args, device):
    model_dict = {
        "PointNet": PointNet,
        "Graph_UNet": Graph_UNet,
        "GraphSAGE": GraphSAGE,
        "MWT": MWT,
        "ONO": ONO,
        "F_FNO": F_FNO,
        "U_FNO": U_FNO,
        "U_NO": U_NO,
        "GNOT": GNOT,
        "Galerkin_Transformer": Galerkin_Transformer,
        "Swin_Transformer": Swin_Transformer,
        "Factformer": Factformer,
        "Transformer": Transformer,
        "U_Net": U_Net,
        "FNO": FNO,
        "Transolver": Transolver,
        "LSM": LSM,
        "DeepONet": DeepONet,
        "MeshGraphNet": MeshGraphNet,
        "RegDGCNN": RegDGCNN,
        "GFNO": GFNO,
    }
    return model_dict[args.model].Model(args, device)
