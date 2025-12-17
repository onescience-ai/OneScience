"""
pix2pix_trainer.py:
- This is a completely original code for the core parts of model training and validation.
- It calls the loss_function, which is also a fully original code, reproducing all the loss functions needed for model training.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nowcasting.loss_function.pool_loss import MaxPoolLoss
from nowcasting.loss_function.evo_loss import EvolutionLoss
from nowcasting.loss_function.csi_loss import CSILoss
import torch.distributed as dist
import os

class Pix2PixTrainer():
    def __init__(self, configs, evolutionnet, generator, discriminator):
        self.configs = configs

        if configs.evo_pre:
            self.evolutionnet = evolutionnet
            self.evo_loss = EvolutionLoss(configs).to(configs.device)
            self.optimizer_E = optim.Adam(evolutionnet.network.parameters(), lr=configs.evo_lr,
                                          betas=(configs.beta1, configs.beta2))
        else:
            self.generator = generator
            self.discriminator = discriminator
            self.criterion = nn.BCEWithLogitsLoss().to(configs.device)
            self.maxpool_loss = MaxPoolLoss(configs).to(configs.device)
            self.csi_loss = CSILoss(configs).to(configs.device)

            state_dict = torch.load(os.path.join(configs.checkpoints_dir, '100_net_evolution_best.ckpt'))
            new_state_dict = self.generator.network.state_dict()
            new_state_dict.update(state_dict) 
            self.generator.network.load_state_dict(new_state_dict)

            if not configs.gen_pre:
                state_dict = torch.load(os.path.join(configs.checkpoints_dir, '100_net_generator_pre_best.ckpt'))
                gen_state_dict = self.generator.network.state_dict()
                gen_state_dict.update(state_dict)
                self.generator.network.load_state_dict(gen_state_dict)

            ''' Ensure that the parameters of 'evo_net' do not participate in gradient updates. '''
            # for name, param in generator.network.named_parameters():
            #     if 'evo_net' in name:
            #         param.requires_grad = False
            # # for name, param in generator.network.named_parameters():
            # #     print(name, param.requires_grad)
            # self.optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, generator.network.parameters()),
            #                               lr=configs.gen_lr, betas=(configs.beta1, configs.beta2))

            self.optimizer_G = optim.Adam(generator.network.parameters(), lr=configs.gen_lr,
                                          betas=(configs.beta1, configs.beta2))

            self.optimizer_D = optim.Adam(discriminator.network.parameters(), lr=configs.dis_lr,
                                          betas=(configs.beta1, configs.beta2))

            self.generator = generator
            self.discriminator = discriminator

    def run_evolution_one_step(self, train_ims):

        real = train_ims[:, self.configs.input_length:, :, :, 0]/128
        real = torch.FloatTensor(real).to(self.configs.device)

        self.optimizer_E.zero_grad()
        evo_result, evo_motion, motion_ = self.evolutionnet.train(train_ims)
        e_loss = self.evo_loss(evo_result, evo_motion, real, motion_[:, :, 0], motion_[:, :, 1])
        e_loss.backward()
        self.optimizer_E.step()

        if dist.is_initialized():
            dist.all_reduce(e_loss)

        return e_loss, evo_result

    def val_evolution_one_step(self, val_ims):
        real = val_ims[:, self.configs.input_length:, :, :, 0]/128
        real = torch.FloatTensor(real).to(self.configs.device)

        evo_result, evo_motion, motion_ = self.evolutionnet.valid(val_ims)
        e_loss = self.evo_loss(evo_result, evo_motion, real, motion_[:, :, 0], motion_[:, :, 1])

        if dist.is_initialized():
            dist.all_reduce(e_loss)

        return e_loss

    def run_generator_one_step(self, train_ims):

        real = train_ims[:, self.configs.input_length:, :, :, 0]
        real = torch.FloatTensor(real).to(self.configs.device)
        real_his = train_ims[:, :self.configs.input_length, :, :, 0]
        real_his = torch.FloatTensor(real_his).to(self.configs.device)

        self.optimizer_G.zero_grad()
        fake = self.generator.train(train_ims)
        fake_pred = self.discriminator.train(torch.cat((real_his, fake), dim=1))

        g_loss = self.criterion(fake_pred, torch.ones_like(fake_pred).to(self.configs.device))
        p_loss = self.maxpool_loss(real, fake)
        # c_loss = (self.csi_loss(real[:,:3], fake[:,:3], threshold = 16) +
        #           self.csi_loss(real[:,:3], fake[:,:3], threshold = 32) +
        #           self.csi_loss(real[:,:3], fake[:,:3], threshold = 64))
        c_loss = self.csi_loss(real, fake, threshold = 16)

        world_rank = dist.get_rank() if dist.is_initialized() else 0
        if world_rank == 0:
            print('g_loss:',g_loss,'p_loss',p_loss,'c_loss',c_loss)

        if self.configs.gen_pre:
            g_loss = p_loss
        else:
            g_loss = g_loss * 6 + p_loss * 20

        g_loss.backward()
        self.optimizer_G.step()

        if dist.is_initialized():
            dist.all_reduce(g_loss)

        return g_loss
    
    def val_generator_one_step(self, val_ims):
        real = val_ims[:, self.configs.input_length:, :, :, 0]
        real = torch.FloatTensor(real).to(self.configs.device)
        real_his = val_ims[:, :self.configs.input_length, :, :, 0]
        real_his = torch.FloatTensor(real_his).to(self.configs.device)

        fake = self.generator.valid(val_ims)
        fake_pred = self.discriminator.valid(torch.cat((real_his, fake), dim=1))
        g_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        p_loss = self.maxpool_loss(real, fake)
        # c_loss = (self.csi_loss(real[:,:3], fake[:,:3], threshold = 16) +
        #           self.csi_loss(real[:,:3], fake[:,:3], threshold = 32) +
        #           self.csi_loss(real[:,:3], fake[:,:3], threshold = 64))
        c_loss = self.csi_loss(real, fake, threshold=16)

        world_rank = dist.get_rank() if dist.is_initialized() else 0
        if world_rank == 0:
            print('g_loss:',g_loss,'p_loss',p_loss,'c_loss',c_loss)

        if self.configs.gen_pre:
            g_loss = p_loss
        else:
            g_loss = g_loss * 6 + p_loss * 20

        if dist.is_initialized():
            dist.all_reduce(g_loss)

        return g_loss

    def run_discriminator_one_step(self, train_ims):

        real = train_ims[:, self.configs.input_length:, :, :, 0]
        real = torch.FloatTensor(real).to(self.configs.device)
        real_his = train_ims[:, :self.configs.input_length, :, :, 0]
        real_his = torch.FloatTensor(real_his).to(self.configs.device)

        self.optimizer_D.zero_grad()
        fake = self.generator.train(train_ims)
        fake_pred = self.discriminator.train(torch.cat((real_his,fake.detach()), dim=1))
        real_pred = self.discriminator.train(torch.cat((real_his,real),dim=1))

        d_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred).to(self.configs.device)) \
                 + self.criterion(real_pred, torch.ones_like(real_pred).to(self.configs.device))
        d_loss.backward()
        self.optimizer_D.step()

        if dist.is_initialized():
            dist.all_reduce(d_loss)

        return d_loss
    
    def val_discriminator_one_step(self, val_ims):
        real = val_ims[:, self.configs.input_length:, :, :, 0]
        real = torch.FloatTensor(real).to(self.configs.device)
        real_his = val_ims[:, :self.configs.input_length, :, :, 0]
        real_his = torch.FloatTensor(real_his).to(self.configs.device)
        
        fake = self.generator.valid(val_ims)
        fake_pred = self.discriminator.valid(torch.cat((real_his, fake.detach()), dim=1))
        real_pred = self.discriminator.valid(torch.cat((real_his, real), dim=1))
        d_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred)) + self.criterion(real_pred, torch.ones_like(real_pred))

        if dist.is_initialized():
            dist.all_reduce(d_loss)
        
        return d_loss



