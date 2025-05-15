from __future__ import absolute_import, division, print_function

import nibabel as nib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
import random
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from trackers import Tracker

from networks import LSDMTrack, LSDM, TrackNet, LSDMTrack_ShareBB
from losses import BalancedLoss, MorphLoss_short, MorphLoss_long, DiceLoss, GradLoss_Long, GradLoss_Short

from lsdmutils import init_weights, read_image, show_image, load_pretrain, get_logger, crop, crop_and_resize, randomstretch, centercrop, randomcrop
from datasets import Pair, CLUSTDataset
from transforms import SiamFCTransforms

from config import config
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel

__all__ = ['LSDMTracker']


class LSDMTracker(Tracker):

    def __init__(self, model_path=None, cfg=None, group=2):
        super(LSDMTracker, self).__init__(model_path, True)

        self.cfg = config

        if cfg:
            config.update(cfg)
            # setup model

        self.net = LSDMTrack_ShareBB(in_channel=9, out_channel=2)

        init_weights(self.net)


        if model_path is not None:
            self.net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        self.tb_writer = SummaryWriter(log_dir="path/to/log/folder")
        self.net = self.net.cuda()
        
        # setup criterio
        self.morph_loss_short = MorphLoss_short()
        self.morph_loss_long = MorphLoss_long()
        self.gradient_loss_long = GradLoss_Long()
        self.gradient_loss_short = GradLoss_Short()
        self.dice = DiceLoss()
        self.track_loss = BalancedLoss()
        self.tags = ["Track_Loss", "LMorph_loss", "SMorph_Loss", "LGrad_Loss", "SGrad_Loss", "Total_Loss", "LR"]

        self.optimizer = optim.SGD(self.net.parameters(), lr=config.initial_lr)

        gamma = np.power(config.ultimate_lr / config.initial_lr, 1.0 / config.epoch_num)

        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

    @torch.no_grad()  #
    def init(self, img, box):
        self.net.eval()

        self.target_sz = box[2:].numpy()
        center= box[:2].numpy()
        self.center = [center[1], center[0]]

        self.upscale_sz = 256
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz)
            )
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes  config.context=1/2
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        
        self.x_sz = self.z_sz * self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = tuple([int(x) for x in np.mean(img, axis=(0, 1))])
        z = crop_and_resize(img, (self.center[0], self.center[1]), self.z_sz,
                            out_size=config.exemplar_sz,
                            border_value=self.avg_color)

        z = np.tile(z, (3))

        # exemplar features
        self.z = torch.from_numpy(z).cuda().permute(2, 0, 1).unsqueeze(0).float()


    @torch.no_grad()  #
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=config.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float()

        # responses
        x = self.net.features(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:config.scale_num // 2] *= config.scale_penalty
        responses[config.scale_num // 2 + 1:] *= config.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - config.window_influence) * response + \
                   config.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           config.total_stride / config.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / config.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - config.scale_lr) * 1.0 + config.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box  [x,y,w,h]
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = read_image(img_file)
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :])

        return boxes, times

    def train_step(self, iter, batch, logger, backward=True):
        # set network mode
        self.net.train(backward)

        t1_frame = batch[3]
        t_frame = batch[4]

        box_t1 = batch[5]
        box_t = batch[6]

        box_t1 = box_t1.type(torch.int32)
        box_t = box_t.type(torch.int32)

        t1_center_y = box_t1[:, 0]
        t1_center_x = box_t1[:, 1]
        t1_h = box_t1[:, 2]
        t1_w = box_t1[:, 3]


        t_center_y = box_t[:, 0]
        t_center_x = box_t[:, 1]
        t_h = box_t[:, 2]
        t_w = box_t[:, 3]


        z_patch_batch = []
        x_patch_batch = []

        nb, c, h, w = t1_frame.shape
        for i in range(nb):

            z_patch_y = t1_center_y[i]
            z_patch_x = t1_center_x[i]
            z_patch_h = t1_h[i]
            z_patch_w = t1_w[i]
            


            z = crop(np.float32(t1_frame[i, :, :, :]), (z_patch_x, z_patch_y, z_patch_h, z_patch_w), self.cfg.instance_sz)
            z = randomstretch(z)
            z = centercrop(z, self.cfg.instance_sz)
            z = randomcrop(z, self.cfg.instance_sz - 8)
            z_patch = centercrop(z, self.cfg.exemplar_sz)

            x_y = t_center_y[i]
            x_x = t_center_x[i]
            x_patch_h = t_h[i]
            x_patch_w = t_w[i]

            x = crop(np.float32(t_frame[i, :, :, :]), (x_x, x_y, x_patch_h, x_patch_w), self.cfg.instance_sz + 8)
            x = randomstretch(x)
            x = centercrop(x, self.cfg.instance_sz + 8)
            x_patch = randomcrop(x, self.cfg.instance_sz)

            x_patch = np.tile(x_patch, (3))
            z_patch = np.tile(z_patch, (3))

            x_patch_batch.append(x_patch)
            z_patch_batch.append(z_patch)

        x_pb = np.array(x_patch_batch)
        z_pb = np.array(z_patch_batch)

        x_pb = torch.from_numpy(x_pb).permute(0, 3, 1, 2).cuda()
        z_pb = torch.from_numpy(z_pb).permute(0, 3, 1, 2).cuda()

        init = batch[0].cuda() # 0
        t1 = batch[1].cuda()  #t-1
        t = batch[2].cuda() #t

        with torch.set_grad_enabled(backward):
            
            long_deformation, short_deformation, registered_init_t1, registered_t1_t, respones = self.net(init, t1, t, z_pb, x_pb)

            registered_init_t1 = registered_init_t1.permute(0, 2, 3, 1)
            registered_t1_t = registered_t1_t.permute(0, 2, 3, 1)

            long_morph_loss = self.morph_loss_long(registered_init_t1, t1)
            short_morph_loss = self.morph_loss_short(registered_t1_t, t)

            grad_loss_long = self.gradient_loss_long(long_deformation)
            grad_loss_short = self.gradient_loss_short(short_deformation)

            # calculate loss
            labels = self._create_labels(respones.size())
            track_loss = self.track_loss(respones, labels)
            logger.info('Track Loss: {}, Long Loss: {}, Short Loss {}, GradLong: {}, GradShort: {}'.format(track_loss, long_morph_loss, short_morph_loss, grad_loss_long, grad_loss_short))

            loss = 1.5 * track_loss + 0.2 * long_morph_loss + 0.2 * short_morph_loss + grad_loss_long + grad_loss_short

            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return track_loss.item(), long_morph_loss.item(), short_morph_loss.item(), grad_loss_long.item(), grad_loss_short.item(), loss.item()

    @torch.enable_grad()
    def train_over(self, list_root, val_seqs=None, save_dir='models'):
        # set to train mode
        logger = get_logger('./models/logs/train_log.log')
        logger.info('start training!')
        self.net.train()
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=config.exemplar_sz,
            instance_sz=config.instance_sz,
            context=config.context)
        
        # loop over epochs
        for epoch in range(config.epoch_num):
            self.lr_scheduler.step(epoch=epoch)
            data = []

            for line in open(list_root):
                data.append(line)
            random.shuffle(data)
            for line_iter, line in enumerate(data):
                track_loss, lm_loss, sm_loss, lg_loss, sg_loss, loss = 0, 0, 0, 0, 0, 0
                logger.info('Patient : {}'.format(line) )
                clust_dataset = CLUSTDataset(line, transforms=transforms)
                clust_dataloader = DataLoader(clust_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             pin_memory=True,
                                             drop_last=True)
                for iter, batch in enumerate(clust_dataloader):
                    loss_1, loss_2, loss_3, loss_4, loss_5, total_loss = self.train_step(iter, batch, logger, backward=True)

                    track_loss += loss_1
                    lm_loss += loss_2
                    sm_loss += loss_3
                    lg_loss += loss_4
                    sg_loss += loss_5
                    loss += total_loss
                    logger.info('Epoch: [{}], Patient:[{}/{}], Progress:[{}/{}] Loss: {:.5f}'.format(epoch + 1, line_iter+1, len(data), iter + 1, len(clust_dataloader), total_loss))
                    sys.stdout.flush()
                track_loss /= len(clust_dataloader)
                lm_loss /= len(clust_dataloader)
                sm_loss /= len(clust_dataloader)
                lg_loss /= len(clust_dataloader)
                sg_loss /= len(clust_dataloader)
                loss /= len(clust_dataloader)
                self.tb_writer.add_scalar(self.tags[0], track_loss, epoch * len(data) + line_iter)
                self.tb_writer.add_scalar(self.tags[1], lm_loss, epoch * len(data) + line_iter)
                self.tb_writer.add_scalar(self.tags[2], sm_loss, epoch * len(data) + line_iter)
                self.tb_writer.add_scalar(self.tags[3], lg_loss, epoch * len(data) + line_iter)
                self.tb_writer.add_scalar(self.tags[4], sg_loss, epoch * len(data) + line_iter)
                self.tb_writer.add_scalar(self.tags[5], loss, epoch * len(data) + line_iter)
                self.tb_writer.add_scalar(self.tags[6], self.optimizer.param_groups[0]["lr"], epoch * len(data) + line_iter)
                print("epoch * len(data) + line_iter :", epoch * len(data) + line_iter)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if epoch % 4 == 0:
                path = os.path.join(save_dir, 'lsdm_%d.pth' % (epoch +1))
                torch.save(self.net.state_dict(), path)

        self.tb_writer.close()

    @torch.no_grad()
    def test_step(self, test_batch):

        t_frame = torch.squeeze(test_batch[4])

        x_patch = [crop_and_resize(np.float32(t_frame[:,:,:]), (self.center[0], self.center[1]), self.x_sz * f, out_size=self.cfg.instance_sz, border_value=self.avg_color) for f in self.scale_factors]
        x_patch = np.stack(x_patch, axis=0)
        
        x_patch = np.tile(x_patch, (3))

        x_pb = np.array(x_patch)
        
        x_pb = torch.from_numpy(x_pb).permute(0, 3, 1, 2).cuda()
        
        init = test_batch[0].cuda()  # 0
        t1 = test_batch[1].cuda()  # t-1
        t = test_batch[2].cuda()  # t

        long_deformation, short_deformation, registered_init_t1, registered_t1_t, responses = self.net(init, t1, t, self.z, x_pb)

        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + self.cfg.window_influence * self.hann_window

        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz
        
        self.center += disp_in_image


        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        box = np.array([self.center[1], self.center[0]])
        
        return box, long_deformation, short_deformation, registered_init_t1, registered_t1_t, init

    @torch.no_grad()
    def test_over(self, clust_dataloader, patient_id, marker_id, result_dir, report_dir):
        self.net.eval()
        patient_result = os.path.join(result_dir, patient_id)
        marker_folder = os.path.join(patient_result, marker_id)
        registered_init_t1_folder = os.path.join(marker_folder, "registered_init_t1")
        registered_t1_t_folder = os.path.join(marker_folder, "registered_t1_t")
        long_morph_folder = os.path.join(marker_folder, "long_morph")
        short_morph_folder = os.path.join(marker_folder, "short_morph")

        if not os.path.exists(patient_result):
            os.mkdir(patient_result)

        if not os.path.exists(marker_folder):
            os.mkdir(marker_folder)
            os.mkdir(registered_init_t1_folder)
            os.mkdir(registered_t1_t_folder)
            os.mkdir(long_morph_folder)
            os.mkdir(short_morph_folder)
        frame_num = len(clust_dataloader)
        box_list = np.zeros((frame_num, 2))
        time_list = []
        for iter, batch in enumerate(clust_dataloader):
            begin = time.time()
            if iter == 0:
                box = torch.squeeze(batch[5])
                init_frame = torch.squeeze(batch[3])
                self.init(np.float32(init_frame), box)
                box_list[iter, :] = torch.squeeze(batch[5])[:2].cpu().numpy()
            else:
                box, long_deformation, short_deformation, registered_init_t1, registered_t1_t, init = self.test_step(test_batch=batch)
                box_list[iter, :] = box
                print("box :", box)
                ldf = long_deformation.cpu().permute(0, 2, 3, 1).squeeze().numpy()
                sdf = short_deformation.cpu().permute(0, 2, 3, 1).squeeze().numpy()

                registered_init_t1 = registered_init_t1.cpu().permute(0, 2, 3, 1).squeeze().numpy()
                registered_t1_t = registered_t1_t.cpu().permute(0, 2, 3, 1).squeeze().numpy()

            time_pimg = time.time() - begin

            time_list.append(time_pimg)
        np.savetxt(os.path.join(patient_result, "{}.txt".format(marker_id)), box_list, fmt='%.3f', delimiter=',')
        np.savetxt(os.path.join(patient_result, "{}_time.txt".format(marker_id)), time_list)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance

            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg, np.ones_like(x) * 0.5, np.zeros_like(x)))

            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = config.r_pos / config.total_stride
        r_neg = config.r_neg / config.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).cuda().float()

        return self.labels