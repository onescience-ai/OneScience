import torch.distributed as dist
import torch.distributed.checkpoint as dcp

def save_checkpoint_torch(rank, model, optimizer, checkpoint_dir):
    # 初始化分布式（假设已 done）
    # dist.init_process_group(backend="nccl", init_method=..., world_size=world_size, rank=rank)

    # 准备 state_dict
    state_dict = {
        "model": model.state_dict(),
        #"optim": optimizer.state_dict()
    }

    # 创建 writer
    writer = dcp.FileSystemWriter(checkpoint_dir)

    # 保存
    dcp.save(
        state_dict=state_dict,
        storage_writer=writer,
        planner=None,       # 使用默认 planner
        process_group=dist.group.WORLD
    )

    if rank == 0:
        print(f"Checkpoint saved to {checkpoint_dir}")

def load_checkpoint_torch(rank, model, optimizer, checkpoint_dir):
    # 创建 reader
    reader = dcp.FileSystemReader(checkpoint_dir)

    # 构建 same shape state_dict keys for load
    state_dict = {
        "model": model.state_dict(),
        #"optim": optimizer.state_dict()
    }

    # 加载
    dcp.load_state_dict(
        state_dict=state_dict,
        storage_reader=reader,
        planner=None,
        process_group=dist.group.WORLD
    )

    model.load_state_dict(state_dict["model"])
    #optimizer.load_state_dict(state_dict["optim"])

    if rank == 0:
        print(f"Checkpoint loaded from {checkpoint_dir}")
