import torch
import os
import sys
import numpy as np
import torch.distributed as dist
import logging
import time
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel
from onescience.models.xihe.xihe import Xihe  # 你的 XiHe 模型定义路径
from onescience.datapipes import CMEMSDatapipe
from onescience.utils.YParams import YParams
from onescience.utils.fcn.darcy_loss import LpLoss

# from apex import optimizers


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")


    ## Distributed config init
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    world_rank = 0
    local_rank = 0
    if cfg.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()
    
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    
    datapipe = CMEMSDatapipe(params=cfg_data, distributed=dist.is_initialized()) #config中的配置读取数据
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()

    # Model init
    model = Xihe(config=cfg).to(local_rank) #根据cfg的model构建模型
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr,betas=tuple(cfg.betas),weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, mode='min')
    loss_obj = LpLoss()                        #loss函数


    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    ## Get model params count
    if cfg.world_size == 1:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n\n")
        print("-" * 50)
        print(f"📂 now params is {total_params}, {total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B")
        print("-" * 50, "\n")

    ## Load model weight if there exist well-trained model 保证可以断点重续
    if os.path.exists(f"{cfg.checkpoint_dir}/model_bak.pth"):
        if world_rank == 0:
            print("\n\n")
            print("-" * 50)
            print(f"✅ There has a model weight, load and continue training...")
            print(f'If you want to train a new model, ensure there is no *.pth file in {cfg.checkpoint_dir}')
            print("-" * 50, "\n")
        ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location=f'cuda:{local_rank}', weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_valid_loss = ckpt["best_valid_loss"]
        best_loss_epoch = ckpt["best_loss_epoch"]
        train_losses = np.load(train_loss_file)
        valid_losses = np.load(valid_loss_file)

    ## Distributed model 并行训练
    if cfg.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    world_rank == 0 and logger.info(f"start training ...")
    
    for epoch in range(cfg.max_epoch):
        print("epoch:",cfg.max_epoch)
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        model.train()        #训练模型
        train_loss = 0
        start_time = time.time()
        for j, data in enumerate(train_dataloader):
            # print("一个epoch的batch数量 =", len(train_dataloader))
            invar = data[0].to(local_rank, dtype=torch.float32)
            outvar = data[1].to(local_rank, dtype=torch.float32)  #做对比的真实数据
            outvar_pred = model(invar)           #输出的预测数据
            
            # if torch.isnan(outvar).any():
            #     print("❌ outvar 中包含 NaN！")
            # else:
            #     print("✅ outvar 正常")
            # # --- 检查 outvar_pred (预测值) ---
            # if torch.isnan(outvar_pred).any():
            #     print("❌ outvar_pred 中包含 NaN！")
            # else:
            #     print("✅ outvar_pred 正常")               
                
            loss = loss_obj(outvar, outvar_pred) #loss计算
            optimizer.zero_grad()                #梯度归0
            loss.backward()                      #反向传播
            optimizer.step()                     #优化器
            train_loss += loss.item()           
            if world_rank == 0:
                logger.info(f'Train: Epoch {epoch}-{j+1}/{len(train_dataloader)} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/(j+1): .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'loss:{train_loss / (j+1): .04f}')

        train_loss /= len(train_dataloader) 

        model.eval() #关闭 dropout（不丢弃）
        valid_loss = 0
        with torch.no_grad():
            start_time = time.time()
            for j, data in enumerate(val_dataloader):
                invar = data[0].to(local_rank, dtype=torch.float32)
                outvar = data[1].to(local_rank, dtype=torch.float32)
                # invar = invar[:, :, :-1, :]
                # outvar = outvar[:, :, :-1, :]
                outvar_pred = model(invar)
                loss = loss_obj(outvar, outvar_pred)

                if cfg.world_size > 1:
                    loss_tensor = loss.detach().to(local_rank) # torch.tensor(loss, device=local_rank)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                    valid_loss += loss
                else:
                    valid_loss += loss.item()   
                if world_rank == 0:
                    logger.info(f'Valid: Epoch {epoch}-{j+1}/{len(val_dataloader)} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/(j+1): .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'loss:{valid_loss / (j+1): .04f}')

        valid_loss /= len(val_dataloader)
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            world_rank == 0 and save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, cfg.checkpoint_dir)
            is_save_ckp = True
        scheduler.step(valid_loss)

        if world_rank == 0:
            logger.info(f"Epoch [{epoch + 1}/{cfg.max_epoch}], "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Valid Loss: {valid_loss:.4f}, "
                        f"Best loss at Epoch: {best_loss_epoch + 1}"
                        + (", saving checkpoint" if is_save_ckp else "")
                        )
            train_losses = np.append(train_losses, train_loss)
            valid_losses = np.append(valid_losses, valid_loss)
            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)

        if epoch - best_loss_epoch > cfg.patience:
            print(f"Loss has not decrease in {cfg.patience} epochs, stopping training...")
            exit()


def save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {"model_state_dict": model_to_save.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(),
             "best_valid_loss": best_valid_loss,
             "best_loss_epoch": best_loss_epoch,
            }
    torch.save(state, f"{model_path}/model.pth")
    ### the weight file saving may interrupted due to DCU queue limit, get a backup to ensure there at least has one model
    os.system(f"mv {model_path}/model.pth {model_path}/model_bak.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
