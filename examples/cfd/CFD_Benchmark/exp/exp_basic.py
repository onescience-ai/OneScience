import torch
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.cfd_benchmark.data_factory import get_data
from onescience.distributed.manager import DistributedManager
from onescience.models.cfd_benchmark.model_factory import get_model


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


class Exp_Basic(object):
    def __init__(self, args):
        self.dist = DistributedManager()
        self.args = args
        self.dataset, self.train_loader, self.test_loader, args.shapelist = get_data(
            args, self.dist
        )
        if self.dist.world_size == 1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.gpu}")
            print(f"Use GPU: cuda:{args.gpu}")
        elif self.dist.world_size > 1 and torch.cuda.is_available():
            self.device = self.dist.device
            print(f"Use DDP: {self.dist.device}")
        else:
            self.device = torch.device("cpu")
            print("Use CPU")

        self.model = get_model(
            args, self.device).to(self.device)
        assert sum(p.numel()
                   for p in self.model.parameters()) > 0, "模型参数为空！"
        if hasattr(self.dataset, "x_normalizer"):
            self.dataset.x_normalizer = self.dataset.x_normalizer.to(
                self.device)
        if hasattr(self.dataset, "y_normalizer"):
            self.dataset.y_normalizer = self.dataset.y_normalizer.to(
                self.device)

        if self.dist.world_size > 1:
            self.dist.find_unused_parameters = args.find_unused_parameters
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.local_rank,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )
            if self.args.use_checkpoint:
                self.model._set_static_graph()
            original_collate_fn = (
                self.train_loader.collate_fn
                if hasattr(self.train_loader, "collate_fn")
                else None
            )
            self.train_sampler = DistributedSampler(
                self.train_loader.dataset,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=True,
            )
            self.train_loader = torch.utils.data.DataLoader(
                self.train_loader.dataset,
                batch_size=args.batch_size,
                sampler=self.train_sampler,
                num_workers=4,
                pin_memory=True,
                collate_fn=original_collate_fn,
            )
        if self.dist.rank == 0:
            print(self.args)
            print(self.model)
            count_parameters(self.model)

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
