import argparse
import os
import time
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import gc
from models.pytorch_i3d_new import InceptionI3d
from models.I3DWSDDA import I3D_WSDDA
from train import train
from val import validate
from test import Test
import logging
import utils
import matplotlib.pyplot as plt
from utils.parser import parse_configuration
import numpy as np
from models.orig_cam import CAM
from models.tsav import TwoStreamAuralVisualModel
import sys
from datasets.dataset_new import ImageList
from datasets.dataset_val import ImageList_val
from datasets.dataset_test import ImageList_test
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from losses.CCC import CCC
#from losses.CCCLoss import CCCLoss
from losses.loss import CCCLoss
from torch.utils.tensorboard import SummaryWriter
#import wandb
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#wandb.init(settings=wandb.Settings(start_method="fork"), project='Audio Visual Fusion')

args = argparse.ArgumentParser(description='DomainAdaptation')
args.add_argument('-c', '--config', default=None, type=str,
					  help='config file path (default: None)')
args = args.parse_args()
configuration = parse_configuration(args.config)

best_Val_acc = 0  # best PrivateTest accuracy
#best_Val_acc = 0  # best PrivateTest accuracy
best_Val_acc_epoch = 0
start_epoch = configuration['model_params']['start_epoch'] #0  # start from epoch 0 or last checkpoint epoch
total_epoch = configuration['model_params']['max_epochs'] #0  # start from epoch 0 or last checkpoint epoch

TrainingAccuracy_V = []
TrainingAccuracy_A = []
ValidationAccuracy_V = []
ValidationAccuracy_A = []

Logfile_name = "LogFiles/" + "log_file.log"
logging.basicConfig(filename=Logfile_name, level=logging.INFO)

tb = SummaryWriter()

SEED = configuration['SEED']
### Using seed for deterministic perfromVisual_model_withI3Dg order
if (SEED == 0):
	cudnn.benchmark = True
else:
	print("Using SEED")
	random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(SEED)

class TrainPadSequence:
	def __call__(self, sorted_batch):
		sequences = [x[0] for x in sorted_batch]
		aud_sequences = [x[1] for x in sorted_batch]
		spec_dim = []

		for aud in aud_sequences:
			spec_dim.append(aud.shape[3])

		max_spec_dim = max(spec_dim)
		audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
		for batch_idx, spectrogram in enumerate(aud_sequences):
			if spectrogram.shape[2] < max_spec_dim:
				audio_features[batch_idx, :, :, :, -spectrogram.shape[3]:] = spectrogram
			else:
				audio_features[batch_idx, :,:, :, :] = spectrogram

		labelV = [x[2] for x in sorted_batch]
		labelA = [x[3] for x in sorted_batch]
		visual_sequences = torch.stack(sequences)
		labelsV = torch.stack(labelV)
		labelsA = torch.stack(labelA)

		return visual_sequences, audio_features, labelsV, labelsA


class ValPadSequence:
	def __call__(self, sorted_batch):

		sequences = [x[0] for x in sorted_batch]
		aud_sequences = [x[1] for x in sorted_batch]
		spec_dim = []
		for aud in aud_sequences:
			spec_dim.append(aud.shape[3])

		max_spec_dim = max(spec_dim)
		audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
		for batch_idx, spectrogram in enumerate(aud_sequences):
			if spectrogram.shape[2] < max_spec_dim:
				audio_features[batch_idx, :, :, :, -spectrogram.shape[3]:] = spectrogram
			else:
				audio_features[batch_idx, :,:, :, :] = spectrogram

		frameids = [x[2] for x in sorted_batch]
		v_ids = [x[3] for x in sorted_batch]
		v_lengths = [x[4] for x in sorted_batch]
		labelV = [x[5] for x in sorted_batch]
		labelA = [x[6] for x in sorted_batch]

		visual_sequences = torch.stack(sequences)
		labelsV = torch.stack(labelV)
		labelsA = torch.stack(labelA)
		return visual_sequences, audio_features, frameids, v_ids, v_lengths, labelsV, labelsA


class TestPadSequence:
	def __call__(self, sorted_batch):

		sequences = [x[0] for x in sorted_batch]
		aud_sequences = [x[1] for x in sorted_batch]
		spec_dim = []
		for aud in aud_sequences:
			spec_dim.append(aud.shape[3])

		max_spec_dim = max(spec_dim)
		audio_features = torch.zeros(len(spec_dim), 16, 1, 64, max_spec_dim)
		for batch_idx, spectrogram in enumerate(aud_sequences):
			if spectrogram.shape[2] < max_spec_dim:
				audio_features[batch_idx, :, :, :, -spectrogram.shape[3]:] = spectrogram
			else:
				audio_features[batch_idx, :,:, :, :] = spectrogram

		frameids = [x[2] for x in sorted_batch]
		v_ids = [x[3] for x in sorted_batch]
		v_lengths = [x[4] for x in sorted_batch]


		visual_sequences = torch.stack(sequences)

		return visual_sequences, audio_features, frameids, v_ids, v_lengths


if not os.path.isdir("SavedWeights"):
	os.makedirs("SavedWeights", exist_ok=True)

path = "SavedWeights"


### Loading audiovisual model
model_path = '../ABAW2020TNT/aff2model_tntsub4/model2/TSAV_Sub4_544k.pth.tar' # path to the model
model = TwoStreamAuralVisualModel(num_channels=4)
saved_model = torch.load(model_path)
model.load_state_dict(saved_model['state_dict'])
model = model.to('cuda')

new_first_layer = nn.Conv3d(in_channels=3,
					out_channels=model.video_model.r2plus1d.stem[0].out_channels,
					kernel_size=model.video_model.r2plus1d.stem[0].kernel_size,
					stride=model.video_model.r2plus1d.stem[0].stride,
					padding=model.video_model.r2plus1d.stem[0].padding,
					bias=False)

new_first_layer.weight.data = model.video_model.r2plus1d.stem[0].weight.data[:, 0:3]
model.video_model.r2plus1d.stem[0] = new_first_layer

### Freezing the model
for p in model.parameters():
	p.requires_grad = False
for p in model.children():
	p.train(False)

## Fusion model
fusion_model = CAM().cuda()

flag = configuration["Mode"]

if flag == "Testing":
	cam_model_path = 'SavedWeights/Val_model_valence_cnn_lstm_mil_64_new_fd_128.pt' # path to the model
	cam_saved_model = torch.load(cam_model_path)
	fusion_model.load_state_dict(cam_saved_model['net'])
	cammodel_accV = torch.load(cam_model_path)['best_Val_accV']
	cammodel_accA = torch.load(cam_model_path)['best_Val_accA']
	print(cammodel_accV)
	print(cammodel_accA)
	for param in fusion_model.parameters():  # children():
		param.requires_grad = False

print('==> Preparing data..')
label_file = '../../SpeechEmotionRec/ratings_gold_standard/ratings_gold_standard/valence/'


if flag == "Training":
	print("Train Data")
	traindataset = ImageList(root=configuration['dataset_rootpath'], fileList=configuration['train_params']['labelpath'],
							audList=configuration['dataset_wavspath'], length=configuration['train_params']['seq_length'],
							flag='train', stride=configuration['train_params']['stride'], dilation = configuration['train_params']['dilation'],
							subseq_length = configuration['train_params']['subseq_length'])
	trainloader = torch.utils.data.DataLoader(
					traindataset, collate_fn=TrainPadSequence(),
					**configuration['train_params']['loader_params'])
			#batch_size=64, shuffle=True, collate_fn=TrainPadSequence(),
			#num_workers=2, pin_memory=True) #, drop_last = True)

	print("Val Data")
	valdataset = ImageList_val(root=configuration['dataset_rootpath'], fileList=configuration['val_params']['labelpath'],
							audList=configuration['dataset_wavspath'], length=configuration['val_params']['seq_length'],
							flag='val', stride=configuration['val_params']['stride'], dilation = configuration['val_params']['dilation'],
							subseq_length = configuration['val_params']['subseq_length'])
	valloader = torch.utils.data.DataLoader(
					valdataset, collate_fn=ValPadSequence(),
					**configuration['val_params']['loader_params'])
	print("Number of Train samples:" + str(len(traindataset)))
	print("Number of Val samples:" + str(len(valdataset)))
else:
	print("Testing")
	testdataset = ImageList_test(root=configuration['dataset_rootpath'], fileList=configuration['test_params']['labelpath'],
						audList=configuration['dataset_wavspath'], length=configuration['test_params']['seq_length'],
						flag='Test', stride=configuration['test_params']['stride'], dilation = configuration['test_params']['dilation'],
						subseq_length = configuration['test_params']['subseq_length'])

	testloader = torch.utils.data.DataLoader(
				testdataset, collate_fn=TestPadSequence(),
				**configuration['test_params']['loader_params'])
	print("Number of Test samples:" + str(len(testdataset)))
	test_tic = time.time()
	Valid_vacc, Valid_aacc = Test(testloader, model, fusion_model)
	test_toc = time.time()
	print("Test phase took {:.1f} seconds".format(test_toc - test_tic))
	sys.exit()

criterion = CCCLoss(digitize_num=1).cuda()
optimizer = torch.optim.Adam(fusion_model.parameters(),# filter(lambda p: p.requires_grad, multimedia_model.parameters()),
								configuration['model_params']['lr'])

scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

cnt = 0
for epoch in range(start_epoch, total_epoch):
	epoch_tic = time.time()
	#adjust_learning_rate(optimizer, epoch)
	#adjust_learning_rate(optimizer, epoch)

	logging.info("Epoch")
	logging.info(epoch)
	#if cnt == 0:
	# train for one epoch
	train_tic = time.time()
	Training_vacc, Training_aacc = train(trainloader, model, criterion, optimizer, scheduler, epoch, fusion_model)
	train_toc = time.time()
	print("Train phase took {:.1f} seconds".format(train_toc - train_tic))
	logging.info("Train phase took {:.1f} seconds".format(train_toc - train_tic))
	#tb.add_scalar("Train Loss", TrainLoss)
	tb.add_scalar("Training_vacc", Training_vacc)
	tb.add_scalar("Training_aacc", Training_aacc)
	#cnt = cnt + 1
	# evaluate on validation set
	#Training_acc = 0.0
	val_tic = time.time()
	Valid_vacc, Valid_aacc = validate(valloader, model, criterion, epoch, fusion_model)
	val_toc = time.time()
	print("Val phase took {:.1f} seconds".format(val_toc - val_tic))
	logging.info("Val phase took {:.1f} seconds".format(val_toc - val_tic))
	#tb.add_scalar("ValidLoss", ValidLoss)
	tb.add_scalar("Valid_vacc", Valid_vacc)
	tb.add_scalar("Valid_aacc", Valid_aacc)
	gc.collect()
	#Test(PrivateTestloader , original_model, criterion, epoch)
	TrainingAccuracy_V.append(Training_vacc)
	TrainingAccuracy_A.append(Training_aacc)
	ValidationAccuracy_V.append(Valid_vacc)
	ValidationAccuracy_A.append(Valid_aacc)

	logging.info('TrainingAccuracy:')
	logging.info(TrainingAccuracy_V)
	logging.info(TrainingAccuracy_A)

	logging.info('ValidationAccuracy:')
	logging.info(ValidationAccuracy_V)
	logging.info(ValidationAccuracy_A)

	if (Valid_vacc + Valid_aacc) > best_Val_acc:
		print('Saving..')
		print("best_Val_accV: %0.3f" % Valid_vacc)
		print("best_Val_accA: %0.3f" % Valid_aacc)
		state = {
			'net': fusion_model.state_dict() ,
			'best_Val_accV': Valid_vacc,
   			'best_Val_accA': Valid_aacc,
			'best_Val_acc_epoch': epoch,
		}
		if not os.path.isdir(path):
			os.mkdir(path)
		torch.save(state, os.path.join(path,'cam_model.pt'))
		best_Val_acc = Valid_vacc + Valid_aacc
		best_Val_acc_epoch = epoch
	epoch_toc = time.time()
	print("Epoch {}/{} took {:.1f} seconds".format(epoch, total_epoch, epoch_toc - epoch_tic))
tb.close()
#print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
#print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_Val_acc)
print("best_PrivateTest_acc_epoch: %d" % best_Val_acc_epoch)
