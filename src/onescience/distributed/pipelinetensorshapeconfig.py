from typing import List, Optional


class PipelineTensorShapeConfig:
    """
    Pipeline tensor shape configuration.

    Args:
        num_stages (int): Total number of pipeline stages.
        stage_shapes (list): A list of length `num_stages - 1`.
            Each element is a list of shapes representing the tensor shapes 
            transmitted between two adjacent stages. `stage_shapes[i]` represents 
            the shapes sent from `stage_i` to `stage_{i+1}`.
    """

    def __init__(
        self,
        num_stages: int,
        stage_shapes: List[List[List[int]]],
    ):
        assert len(stage_shapes) == num_stages - 1, (
            f"Length of stage_shapes should be num_stages-1={num_stages-1}, "
            f"but got {len(stage_shapes)}"
        )
        self.num_stages = num_stages
        self.stage_shapes = stage_shapes

    def get_shapes(self, pp_rank: int):
        """
        Returns the (recv_tensor_shapes, send_tensor_shapes) for the specified pp_rank.

        Args:
            pp_rank (int): The current pipeline rank.

        Returns:
            tuple: (recv_tensor_shapes, send_tensor_shapes)
                   For the first stage's recv and the last stage's send, the value is [None].
        """
        assert 0 <= pp_rank < self.num_stages, (
            f"pp_rank={pp_rank} is out of range [0, {self.num_stages})"
        )

        # First stage: does not receive, only sends
        if pp_rank == 0:
            recv_tensor_shapes = [None]
            send_tensor_shapes = self.stage_shapes[0]

        # Last stage: only receives, does not send
        elif pp_rank == self.num_stages - 1:
            recv_tensor_shapes = self.stage_shapes[pp_rank - 1]
            send_tensor_shapes = [None]

        # Intermediate stages: receive from the previous stage and send to the next stage
        else:
            recv_tensor_shapes = self.stage_shapes[pp_rank - 1]
            send_tensor_shapes = self.stage_shapes[pp_rank]

        return recv_tensor_shapes, send_tensor_shapes