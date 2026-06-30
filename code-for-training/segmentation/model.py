"""
Model classes to implement U-Net-based segmentation models.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 Gijs van Tulder / TU Delft

import torch
import torch.nn
import torch.nn.functional as F
import torch.autograd
import lightning.pytorch as pl
import torchmetrics
import monai.networks.blocks
import monai.networks.nets
import monai.networks
import monai.losses

models = {}
def register_model(cls):
    models[cls.__name__] = cls
    return cls


class HipSegmentation(pl.LightningModule):
    def __init__(self, lr, num_classes, weight_decay=None, loss_type=['dice', 'ce']):
        super().__init__()
        self.lr = lr
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.build_net()
        self.save_hyperparameters()

    def build_net(self):
        raise NotImplementedError('build_net must be implemented')
        
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=(self.weight_decay or 0)
        )

    def training_step(self, batch, batch_idx):
        return self._compute_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._compute_step('val', batch, batch_idx)

    def forward(self, *x):
        return self.net(*x)

    def predict(self, *x):
        return self.net(*x)

    def _compute_step(self, phase, batch, batch_idx):
        assert phase in ('train', 'val')
        xs, ys, *meta = batch

        # run the network
        preds = self(xs)

        if self.num_classes in (4, 5):
            # five-class task
            # 0: unknown background
            # 1: known background
            # 2: acetabulum
            # 3: femur
            # 4: joint space

            # four-class task
            # 0: unknown background
            # 1: known background
            # 2: femur
            # 3: joint space

            # mask background
            mask = (ys[:, None, :, :] != 0)

            # compute the loss
            loss_dice = monai.losses.MaskedDiceLoss(
                include_background=False,
                softmax=True,
                to_onehot_y=False,
            )(preds, monai.networks.one_hot(ys[:, None, :, :], self.num_classes), mask)
            loss_ce = torch.nn.CrossEntropyLoss(
                ignore_index=0,
            )(preds, ys)
            if 'focal' in self.loss_type:
                loss_focal = monai.losses.FocalLoss(
                    include_background=False,
                    use_softmax=True,
                    to_onehot_y=False,
                )(preds, monai.networks.one_hot(ys[:, None, :, :], self.num_classes))
            if 'hd' in self.loss_type:
                loss_hd = monai.losses.HausdorffDTLoss(
                    include_background=False,
                    softmax=True,
                    to_onehot_y=False,
                )(preds, monai.networks.one_hot(ys[:, None, :, :], self.num_classes))

        elif self.num_classes == 1:
            # binary task
            # 0: background
            # 1: joint space

            # compute the loss
            loss_dice = monai.losses.DiceLoss(
                sigmoid=True,
            )(preds, ys[:, None, :, :])
            loss_ce = torch.nn.functional.binary_cross_entropy_with_logits(
                preds, ys[:, None, :, :].to(preds.dtype)
            )
            assert self.loss_type == ['dice', 'ce']

        else:
            raise ValueError(f'unexpected num_classes: {self.num_classes}')

        # combined loss
        loss = []
        if 'dice' in self.loss_type:
            loss.append(loss_dice)
        if 'ce' in self.loss_type:
            loss.append(loss_ce)
        if 'focal' in self.loss_type:
            loss.append(loss_focal)
        if 'hd' in self.loss_type:
            loss.append(loss_hd)
        loss = sum(loss)

        # collect scores for logging
        scores = {
            f'{phase}/loss': loss,
            f'{phase}/loss_dice': loss_dice,
            f'{phase}/loss_ce': loss_ce,
        }
        if 'focal' in self.loss_type:
            scores[f'{phase}/loss_focal'] = loss_focal
        if 'hd' in self.loss_type:
            scores[f'{phase}/loss_hd'] = loss_hd
        self.log_dict(scores, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


@register_model
class HipSegmentation_UNet_4(HipSegmentation):
    def build_net(self):
        self.net = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=self.num_classes,
            channels=(4, 8, 16, 32),
            strides=(2, 2, 2),
            num_res_units=2
        )


@register_model
class HipSegmentation_UNet_4_Wide2(HipSegmentation):
    def build_net(self):
        self.net = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=self.num_classes,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
            num_res_units=2
        )


@register_model
class HipSegmentation_UNet_4_Wide4(HipSegmentation):
    def build_net(self):
        self.net = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=self.num_classes,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2
        )


@register_model
class HipSegmentation_UNet_4_Wide8(HipSegmentation):
    def build_net(self):
        self.net = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=self.num_classes,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2
        )


@register_model
class HipSegmentation_UNet_4_Wide8_DiceOnly(HipSegmentation_UNet_4_Wide8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, loss_type=['dice'])


@register_model
class HipSegmentation_UNet_4_Wide8_FocalLoss(HipSegmentation_UNet_4_Wide8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, loss_type=['dice', 'focal'])


@register_model
class HipSegmentation_UNet_4_Wide8_HDLoss(HipSegmentation_UNet_4_Wide8):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, loss_type=['hd'])
