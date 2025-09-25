import torch


class register_memory_class:
    """
    这是一个用于注册内存监控的类，它可以在模型的前向和反向传播过程中注册用于内存监控的hook。
    初始化参数:
    - logger: 日志记录器对象，默认为None，即直接print函数打印
    - forward: bool类型变量，表示是否需要监控前向传播过程中的内存使用情况，默认为True
    - backward: bool类型变量，表示是否需要监控反向传播过程中的内存使用情况，默认为True

    使用示例
    cls_ins = register_memory_class(logger)
    cls_ins.apply(model)


    @author: wx
    @date: 2025-01-16
    """

    def __init__(self, logger=None, forward=True, backward=True):
        if logger == None:
            self.logger_flag = False
        else:
            self.logger_flag = True
            self.logger = logger
        self.forward = forward
        self.backward = backward

    def apply(self, model):
        def monitor_memory_forward(module, input, output):
            if self.logger_flag == False:
                print(
                    f"forward Layer: {module.__class__.__name__}, Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
                )
            else:
                self.logger.info(
                    f"forward Layer: {module.__class__.__name__}, Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
                )

        def monitor_memory_backward(module, grad_input, grad_output):
            if self.logger_flag == False:
                print(
                    f"Backward Layer for {module.__class__.__name__}, Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
                )
            else:
                self.logger.info(
                    f"Backward Layer for {module.__class__.__name__}, Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
                )

        if self.forward:
            for name, layer in model.named_modules():
                layer.register_forward_hook(monitor_memory_forward)
        if self.backward:
            for name, layer in model.named_modules():
                layer.register_backward_hook(monitor_memory_backward)
