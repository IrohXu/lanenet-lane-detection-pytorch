# ----------------------------------------
# Written by Yude Wang
# Change by Iroh Cao
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from model.lanenet.backbone.deeplabv3_plus.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from model.lanenet.backbone.deeplabv3_plus.backbone import build_backbone
from model.lanenet.backbone.deeplabv3_plus.ASPP import ASPP

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class Deeplabv3plus_Encoder(nn.Module):
	def __init__(self):
		super(Deeplabv3plus_Encoder, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=256, 
				rate=16//16)
		self.dropout1 = nn.Dropout(0.5)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, 48, 1, 1, padding=1//2,bias=True),
				# SynchronizedBatchNorm2d(48, momentum=0.0003),
				nn.BatchNorm2d(48),
				nn.ReLU(inplace=True),		
		)		

		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		# 	elif isinstance(m, SynchronizedBatchNorm2d):
		# 		nn.init.constant_(m.weight, 1)
		# 		nn.init.constant_(m.bias, 0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				weights_init_kaiming(m)
			elif isinstance(m, nn.BatchNorm2d):
				weights_init_kaiming(m)

		self.backbone = build_backbone('res101_atrous', os=16)
		self.backbone_layers = self.backbone.get_layers()
	
	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_shallow = self.shortcut_conv(layers[0])
		
		return feature_aspp, feature_shallow

class Deeplabv3plus_Decoder(nn.Module):
	def __init__(self, out_dim):
		super(Deeplabv3plus_Decoder, self).__init__()

		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//4)
	
		self.cat_conv = nn.Sequential(
				nn.Conv2d(304, 256, 3, 1, padding=1,bias=True),
				# SynchronizedBatchNorm2d(256, momentum=0.0003),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
				# SynchronizedBatchNorm2d(256, momentum=0.0003),
				nn.BatchNorm2d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, out_dim, 1, 1, padding=0)
		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		# 	elif isinstance(m, SynchronizedBatchNorm2d):
		# 		nn.init.constant_(m.weight, 1)
		# 		nn.init.constant_(m.bias, 0)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				weights_init_kaiming(m)
			elif isinstance(m, nn.BatchNorm2d):
				weights_init_kaiming(m)
	
	def forward(self, feature_aspp, feature_shallow):
    		
		feature_aspp = self.upsample_sub(feature_aspp)
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat) 
		result = self.cls_conv(result)
		result = self.upsample4(result)
		return result

# class deeplabv3plus(nn.Module):
# 	def __init__(self, cfg):
# 		super(deeplabv3plus, self).__init__()
# 		self.backbone = None		
# 		self.backbone_layers = None
# 		input_channel = 2048		
# 		self.aspp = ASPP(dim_in=input_channel, 
# 				dim_out=cfg.MODEL_ASPP_OUTDIM, 
# 				rate=16//cfg.MODEL_OUTPUT_STRIDE,
# 				bn_mom = cfg.TRAIN_BN_MOM)
# 		self.dropout1 = nn.Dropout(0.5)
# 		self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
# 		self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE//4)

# 		indim = 256
# 		self.shortcut_conv = nn.Sequential(
# 				nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1, padding=cfg.MODEL_SHORTCUT_KERNEL//2,bias=True),
# 				SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
# 				nn.ReLU(inplace=True),		
# 		)		
# 		self.cat_conv = nn.Sequential(
# 				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM+cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
# 				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
# 				nn.ReLU(inplace=True),
# 				nn.Dropout(0.5),
# 				nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,bias=True),
# 				SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
# 				nn.ReLU(inplace=True),
# 				nn.Dropout(0.1),
# 		)
# 		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
# 		for m in self.modules():
# 			if isinstance(m, nn.Conv2d):
# 				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
# 			elif isinstance(m, SynchronizedBatchNorm2d):
# 				nn.init.constant_(m.weight, 1)
# 				nn.init.constant_(m.bias, 0)
# 		self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
# 		self.backbone_layers = self.backbone.get_layers()

# 	def forward(self, x):
# 		x_bottom = self.backbone(x)
# 		layers = self.backbone.get_layers()
# 		feature_aspp = self.aspp(layers[-1])
# 		feature_aspp = self.dropout1(feature_aspp)
# 		feature_aspp = self.upsample_sub(feature_aspp)

# 		feature_shallow = self.shortcut_conv(layers[0])
# 		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
# 		result = self.cat_conv(feature_cat) 
# 		result = self.cls_conv(result)
# 		result = self.upsample4(result)
# 		return result

