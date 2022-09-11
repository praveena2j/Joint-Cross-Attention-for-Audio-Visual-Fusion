from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
# import scipy as sp
from scipy import signal
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from utils.utils import clip_gradient

import utils.utils as utils
from utils.exp_utils import pearson
from EvaluationMetrics.ICC import compute_icc
from EvaluationMetrics.cccmetric import ccc

from utils.utils import Normalize
from utils.utils import calc_scores
import logging
# import models.resnet as ResNet
#import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys
import math
from losses.CCC import CCC
#import wandb
learning_rate_decay_start = 5  # 50
learning_rate_decay_every = 2 # 5
learning_rate_decay_rate = 0.8 # 0.9
total_epoch = 30
lr = 0.001
scaler = torch.cuda.amp.GradScaler()

def train(train_loader, model, criterion, optimizer, scheduler, epoch, cam):
	print('\nEpoch: %d' % epoch)
	global Train_acc
	#wandb.watch(audiovisual_model, log_freq=100)
	#wandb.watch(cam, log_freq=100)

	# switch to train mode
	#audiovisual_model.train()
	model.eval()
	cam.train()

	epoch_loss = 0
	correct = 0
	total = 0
	running_loss = 0
	running_accuracy = 0
	vout = list()
	vtar = list()

	aout = list()
	atar = list()

	if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
		frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
		decay_factor = learning_rate_decay_rate ** frac
		current_lr = lr * decay_factor
		utils.set_lr(optimizer, current_lr)  # set the decayed rate
	else:
		current_lr = lr
	print('learning_rate: %s' % str(current_lr))
	logging.info("Learning rate")
	logging.info(current_lr)
	#torch.cuda.synchronize()
	#t1 = time.time()
	n = 0
	for batch_idx, (visualdata, audiodata, labels_V, labels_A) in tqdm(enumerate(train_loader),
				 										 total=len(train_loader), position=0, leave=True):

		#if(batch_idx > 5):#int(65844/64)):
		#	break
		#torch.cuda.synchronize()
		#t2 = time.time()
		#print('data loading time', t2-t1)
		optimizer.zero_grad(set_to_none=True)
		audiodata = audiodata.cuda()#.unsqueeze(2)

		visualdata = visualdata.cuda()#permute(0,4,1,2,3).cuda()
		#labels = labels.cuda()
		#visuallabel = visuallabel.squeeze(1).type(torch.FloatTensor).cuda()
		#print("training started")
		#torch.cuda.synchronize()
		#t3 = time.time()

		with torch.cuda.amp.autocast():
			with torch.no_grad():
				b, seq_t, c, subseq_t, h, w = visualdata.size()

				#visualdata = visual_data.view(b, c, -1, sub_seq_len, h, w)
				visual_feats = torch.empty((b, seq_t, 25088), dtype=visualdata.dtype, device = visualdata.device)
				aud_feats = torch.empty((b, seq_t, 512), dtype=visualdata.dtype, device = visualdata.device)

				#vis_data = visualdata.view(b*visualdata.shape[2], c, subseq_t ,h , w)
				#visualfeatures, _ = visual_model(vis_data)
				#visual_feat = visualfeatures.view(b, -1, visualfeatures.shape[1])
				#print(visfin_test.shape)
				#print(visual_feat.shape)
				for i in range(visualdata.shape[0]):
					#vis_dat = visualdata[i, :, :, :,:,:].transpose(0,1)
					#print(vis_dat.shape)
					#visualfeat = visual_model(visualdata[i, :, :, :,:,:].transpose(0,1))#[:,-1,:]

					aud_feat, visualfeat, _ = model(audiodata[i,:,:,:], visualdata[i, :, :, :,:,:])

					#visualfeat, _ = torch.max(visualfeat,1)
					#print(visualfeat.shape)
					#visual_feat = visualfeat.view(b, -1, visualfeat.shape[1])
					#visual_feats.append(visualfeat)
					visual_feats[i,:,:] = visualfeat
					#aud_data = audiodata.view(audiodata.shape[0]*audiodata.shape[1], audiodata.shape[2], audiodata.shape[3]).unsqueeze(1)
					#aud_data = audiodata[i,:,:,:]#.unsqueeze(1)
					#aud_feat = audio_model(aud_data)
					#print(aud_feat.shape)
					#audio_feat = aud_feat.view(b, -1, aud_feat.shape[1])
					#aud_feats.append(aud_feat) #.squeeze(3))
					aud_feats[i,:,:] = aud_feat
				#print(audio_feat.shape)
				#visual_feat = torch.stack(visual_feats)#.squeeze(3).squeeze(3).squeeze(3)#.transpose(1,2)
				#visual_feat = visual_feat.view(visual_feat.shape[0]*visual_feat.shape[1], -1)
				#print(visual_feat.shape)
				#torch.cuda.synchronize()
				#t4 = time.time()
				#print('visual feature extraction time', t4-t3)

				#torch.cuda.synchronize()
				#t5 = time.time()
				#aud_feats = []
				#print(audiodata.shape)
				#for i in range(audiodata.shape[0]):
				#	aud_data = audiodata[i,:,:,:].unsqueeze(1)
				#	audio_feat, audio_out = audio_model(aud_data)
				#	aud_feats.append(audio_feat.squeeze(3))
				#audio_feat = torch.stack(aud_feats)#.squeeze(3)#.transpose(1,2)
				#print(audio_feat.shape)
				#print(visual_feat.shape)
				#torch.cuda.synchronize()
				#t6 = time.time()
				#print('audio feature extraction time', t6-t5)
				#audio_feat = audio_feat.squeeze(3)#.transpose(1,2)

				#visual_feat = torch.max(visual_feat, dim = 2)[0]#.squeeze(2).squeeze(2)
				#audio_feat = torch.max(audio_feat, dim = 2)[0]#.squeeze(2).squeeze(2)

				#print("features extracted")
				#audio_feat_norm = F.normalize(audio_feat, p=2, dim=2, eps=1e-12)
				#visual_feat_norm = F.normalize(visual_feat, p=2, dim=2, eps=1e-12)

			audiovisual_vouts,audiovisual_aouts = cam(aud_feats, visual_feats)
			#audio_attfeat, visual_attfeat = cam(audio_feat_norm, visual_feat_norm)
			#audiovisual_outs = audiovisual_model(audio_attfeat, visual_attfeat)

			voutputs = audiovisual_vouts.view(-1, audiovisual_vouts.shape[0]*audiovisual_vouts.shape[1])
			aoutputs = audiovisual_aouts.view(-1, audiovisual_aouts.shape[0]*audiovisual_aouts.shape[1])
			vtargets = labels_V.view(-1, labels_V.shape[0]*labels_V.shape[1]).cuda()
			atargets = labels_A.view(-1, labels_A.shape[0]*labels_A.shape[1]).cuda()

			v_loss = criterion(voutputs, vtargets)
			a_loss = criterion(aoutputs, atargets)
			final_loss = v_loss + a_loss
			epoch_loss += final_loss.cpu().data.numpy()
		scaler.scale(final_loss).backward()
		scaler.step(optimizer)
		scaler.update()
		n = n + 1
		#final_loss.backward()
		#optimizer.step()

		vout = vout + voutputs.squeeze(0).detach().cpu().tolist()
		vtar = vtar + vtargets.squeeze(0).detach().cpu().tolist()

		aout = aout + aoutputs.squeeze(0).detach().cpu().tolist()
		atar = atar + atargets.squeeze(0).detach().cpu().tolist()

		#vout = np.concatenate([vout, voutputs.squeeze(0).detach().cpu().numpy()])
		#vtar = np.concatenate([vtar, vtargets.squeeze(0).detach().cpu().numpy()])

		#aout = np.concatenate([aout, aoutputs.squeeze(0).detach().cpu().numpy()])
		#atar = np.concatenate([atar, atargets.squeeze(0).detach().cpu().numpy()])

		#if torch.isnan(loss):
		#	print(outputs)
		#	print(targets)
		#	print(loss)
		#	sys.exit()

		#if batch_idx % 100 == 0:
		#	wandb.log({"train_loss": loss})
		#	pass

	scheduler.step(epoch_loss / n)

	#pred, tar = Normalize(out, tar)

	#flags = vtar == -5.0
	#print(np.array(vout).shape)
	#print(np.squeeze(np.array(vout)).shape)
	#vout = np.delete(np.array(vout), flags)
	#print(len(vout))
	#sys.exit()
	#vtar = np.delete(vtar, flags)
	#aout = np.delete(aout, flags)
	#atar = np.delete(atar, flags)


	if (len(vtar) > 1):
		train_vacc = ccc(vout, vtar)
		train_aacc = ccc(aout, atar)
	else:
		train_acc = 0
	print("Train Accuracy")
	#wandb.log({"train_acc": train_acc})
	print(train_vacc)
	print(train_aacc)
	#xcorr_weights = 0
	return train_vacc, train_aacc
