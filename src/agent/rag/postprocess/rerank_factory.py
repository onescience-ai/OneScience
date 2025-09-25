from agent.rag.postprocess.rerank_base import RerankMode
from agent.rag.postprocess.rerank_model import RerankModelRunner
from agent.rag.postprocess.weight_rerank import WeightRerankRunner


def create_reranker(config: dict):
    reranking_mode = config.get("reranking_mode", "")
    rerank_model_config = config[reranking_mode]
    match reranking_mode:
        case RerankMode.RERANKING_MODEL.value:
            runner = RerankModelRunner(rerank_model_config)
        case RerankMode.WEIGHTED_SCORE.value:
            runner = WeightRerankRunner(rerank_model_config)
        case _:
            raise ValueError(f"Unknown runner type: {reranking_mode}")

    return runner
