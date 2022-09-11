import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
import torchaudio
from torchvision import transforms
import torch
from scipy import signal
from .spec_transform import *
from .clip_transforms import *
import bisect
import cv2
import pandas as pd
import utils.videotransforms as videotransforms
import re
from models.vggish_pytorch import vggish_input
import csv
import math

def get_filename(n):
	filename, ext = os.path.splitext(os.path.basename(n))
	return filename

def default_seq_reader(videoslist, win_length, stride, dilation, wavs_list):
	shift_length = stride #length-1
	sequences = []
	csv_data_list = os.listdir(videoslist)
	skip_vids = ['313.csv', '212.csv', '303.csv', '171.csv', '40-30-1280x720.csv', '286.csv', '270.csv', '234.csv', '239.csv', '266.csv']

	print("Number of Sequences: " + str(len(set(csv_data_list))))
	for video in csv_data_list:
		#video = '10-60-1280x720_right.csv'
		if video.startswith('.'):
			continue
		if video in skip_vids:
			continue
		vid_data = pd.read_csv(os.path.join(videoslist,video))
		video_data = vid_data.to_dict("list")
		images = video_data['img']
		labels_V = video_data['V']
		labels_A = video_data['A']
		label_arrayV = np.asarray(labels_V, dtype = np.float32)
		label_arrayA = np.asarray(labels_A, dtype = np.float32)
		#medfiltered_labels = signal.medfilt(label_array)
		frame_ids = video_data['frame_id']
		#timestamp_file = os.path.join(time_list, get_filename(video) +'_video_ts.txt')
		f_name = get_filename(video)
		if f_name.endswith('_left'):
			wav_file_path = os.path.join(wavs_list, f_name[:-5])
			vidname = f_name[:-5]
		elif f_name.endswith('_right'):
			wav_file_path = os.path.join(wavs_list, f_name[:-6])
			vidname = f_name[:-6]
		else:
			wav_file_path = os.path.join(wavs_list, f_name)
			vidname = f_name
		#label_array = np.asarray(labels_V, dtype=np.float32)
		#medfiltered_labels = signal.medfilt(label_array)
		vid = np.asarray(list(zip(images, label_arrayV, label_arrayA)))
		#f = open(timestamp_file)
		#time_lines = f.readlines()
		frameid_array = np.asarray(frame_ids, dtype=np.int32)
		time_filename = os.path.join('../../Datasets/Affwild2/realtimestamps_orig', vidname) + '_video_ts.txt'
		f = open(os.path.join(time_filename))
		lines = f.readlines()[1:]
		length = len(lines) #len(os.listdir(wav_file_path))
		end = 481
		start = end -win_length
		#start = 0 #end - win_length
		#end = start + win_length
		counter = 0
		cnt = 0
		result = []
		#if end < length:
		while end < length + 481:
			avail_seq_length = end -start
			#sequence_length = win_length / dilation
			# Extracting the indices between the start and start + 128 (sequence length)
			#indices = np.arange(start, end, dilation) + (dilation -1)
			#indices = np.arange(math.ceil(sequence_length))
			#indices = np.flip(end - dilation*(np.arange(math.ceil(sequence_length))))
			#frame_id = frameid_array[indices]
			#print(frame_id)
			#indices = np.where((frameid_array>=start+1) & (frameid_array<=end))[0]
			count = 15
			#subseq_indices_check = []
			num_samples = 0
			vis_subsequnces = []
			aud_subsequnces = []
			for i in range(16):
				#subseq_indices.append(np.where((frameid_array>=((i-1)*32)+1) & (frameid_array<=32*i))[0])
				sub_indices = np.where((frameid_array>=(start+(i*32))+1) & (frameid_array<=(end -(count*32))))[0]
				wav_file = os.path.join(wav_file_path, str(end -(count*32))) +'.wav'
				#print(sub_indices)
				if (end -(count*32)) <= length:
					result.append(end -(count*32))
					#if ((start+(i*32))+1) <0 and (end -(count*32)) <0:
					#	subsequnces.append([])
					if len(sub_indices)>=8 and len(sub_indices)<16:
						subseq_indices = sub_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices)>=16 and len(sub_indices)<24:
						subseq_indices = np.flip(np.flip(sub_indices)[::2])
						subseq_indices = subseq_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices)>=24 and len(sub_indices)<32:
						subseq_indices = np.flip(np.flip(sub_indices)[::3])
						subseq_indices = subseq_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices) == 32:
						subseq_indices = np.flip(np.flip(sub_indices)[::4])
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices) > 0 and len(sub_indices) < 8:
						newList = [sub_indices[-1]]* (8-len(sub_indices))
						sub_indices = np.append(sub_indices, np.array(newList), 0)
						#sub_indices.extend(newList)
						vis_subsequnces.append(vid[sub_indices])
						aud_subsequnces.append(wav_file)
					#else:
					#	subsequnces.append([])
				count = count - 1

			start_frame_id = start +1
			#wav_file = os.path.join(wav_file_path, str(end)) +'.wav'

			if len(vis_subsequnces) == 16:
				#sequences.append([subsequnces, wav_file, start_frame_id])
				sequences.append([vis_subsequnces, aud_subsequnces])
			#else:
			#	sequences.append([])
			if avail_seq_length>512:
				print("Wrong Sequence")
				#sys.exit()
			counter = counter + 1
			if counter > 31:
				end = end + 480 + shift_length
				start = end - win_length
				#start = start + 224 + shift_length
				#end = start + win_length
				counter = 0
			else:
				end = end + shift_length
				start = end - win_length

		result.sort()
		#print(result)
		#print("-----------")
		if len(set(result)) == length:
			continue
		else:
			print(video)
			print(len(set(result)))
			print(length)
		#	#sequences.append([seq, wav_file, start_frame_id])
	return sequences

def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		#print(fileList)
		video_length = 0
		videos = []
		lines = list(file)
		#print(len(lines))
		for i in range(9):
			line = lines[video_length]
			#print(line)
			#line = file.readlines()[video_length + i]
			imgPath, label = line.strip().split(' ')
			find_str = os.path.dirname(imgPath)
			#print(find_str)
			new_video_length = 0
			for line in lines:
				if find_str in line:
					new_video_length = new_video_length + 1
			#print(new_video_length)Visualmodel_for_Afwild2_bestworkingcode_avail_lab_img_videolevel_perf
			videos.append(lines[video_length:video_length + new_video_length])
			video_length = video_length + new_video_length
			#print(video_length)
	return videos

class ImageList(data.Dataset):
	def __init__(self, root, fileList, audList, length, flag, stride, dilation, subseq_length, list_reader=default_list_reader, seq_reader=default_seq_reader):
		self.root = root
		#self.label_path = label_path
		self.videoslist = fileList #list_reader(fileList)
		self.win_length = length
		#self.time_list = timestmps
		self.num_subseqs = int(self.win_length / subseq_length)
		self.wavs_list = audList
		self.stride = stride
		self.dilation = dilation
		self.subseq_length = int(subseq_length / self.dilation)
		self.sequence_list = seq_reader(self.videoslist, self.win_length, self.stride, self.dilation, self.wavs_list)
		#self.stride = stride
		self.sample_rate = 44100
		self.window_size = 20e-3
		self.window_stride = 10e-3
		self.sample_len_secs = 1
		self.sample_len_clipframes = int(self.sample_len_secs * self.sample_rate * self.num_subseqs)
		self.sample_len_frames = int(self.sample_len_secs * self.sample_rate)
		self.audio_shift_sec = 1
		self.audio_shift_samples = int(self.audio_shift_sec * self.sample_rate)

		#self.transform = transform
		#self.dataset = dataset
		#self.loader = loader
		self.flag = flag

	def __getitem__(self, index):
		#for video in self.videoslist:
		seq_path, wav_file = self.sequence_list[index]
		#seq_path = self.sequence_list[index]
		#img = self.loader(os.path.join(self.root, imgPath), self.flag)
		#if (self.flag == 'train'):
		seq, label_V, label_A = self.load_vis_data(self.root, seq_path, self.flag, self.subseq_length)
		aud_data = self.load_aud_data(wav_file, self.num_subseqs, self.flag)
		#label_index = torch.DoubleTensor([label])
		#else:
		#   seq, label = self.load_test_data_label(seq_path)
		#   label_index = torch.LongTensor([label])
		#if self.transform is not None:
		#    img = self.transform(img)
		return seq, aud_data, label_V, label_A#_index

	def __len__(self):
		return len(self.sequence_list)

	def load_vis_data(self, root, SeqPath, flag, subseq_len):
		#print("Loadung training data")
		clip_transform = ComposeWithInvert([NumpyToTensor(),
												 Normalize(mean=[0.43216, 0.394666, 0.37645],
														   std=[0.22803, 0.22145, 0.216989])])
		if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip(),
					#transforms.RandomResizedCrop(224),
					#transforms.RandomHorizontalFlip(),
					#transforms.ToTensor(),
			])
		else:
			data_transforms=transforms.Compose([videotransforms.CenterCrop(224),
				#transforms.Resize(256),
				#transforms.CenterCrop(224),
				#transforms.ToTensor(),
			])
		output = []
		subseq_inputs = []
		subseq_labels = []
		labV = []
		labA = []
		frame_ids = []
		seq_length = math.ceil(self.win_length / self.dilation)
		seqs = []
		for clip in SeqPath:
			images = np.zeros((8, 112, 112, 3), dtype=np.uint8)
			#inputs = []
			labelV = -5.0
			labelA = -5.0
			for im_index, image in enumerate(clip):
				imgPath = image[0]
				labelV = image[1]
				labelA = image[2]

				#try:
				#	img = cv2.imread(root + imgPath)
				#except:
				#	img = cv2.imread(root + imgPath)
				try:
					img = np.array(Image.open(os.path.join(root , imgPath)))
					images[im_index, :, :, 0:3] = img
				except:
					pass

			#imgs = data_transforms(images)
			imgs = clip_transform(RandomColorAugmentation(images))
			seqs.append(imgs)
			#seqs.append(imgs)
			#subseq_inputs.append(inputs)
			labV.append(float(labelV))
			labA.append(float(labelA))

			#lab.append(float(label))
			#if len(inputs) == subseq_len:
			#	subseq_inputs.append(inputs)
			#	lab.append(float(label))
			#	inputs = []

		targetsV = torch.FloatTensor(labV)
		targetsA = torch.FloatTensor(labA)
		#targets = torch.mean(label)
		#for subseq in subseq_inputs:
		#	imgs=np.asarray(subseq, dtype=np.float32)
		#	#if(imgs.shape[0] != 0):
		#	imgs = data_transforms(imgs)
		#	#seqs.append(imgs)
		#	seqs.append(torch.from_numpy(imgs))
		vid_seqs = torch.stack(seqs)#.permute(4,0,1,2,3)
		#vid_seqs = np.stack(seqs)#.permute(4,0,1,2,3)
		return vid_seqs, targetsV, targetsA # vid_seqs,

	def load_aud_data(self, wav_file, num_subseqs, flag):
		transform_spectra = transforms.Compose([
			transforms.ToPILImage(),
			#transforms.Resize((224,224)),
			transforms.RandomVerticalFlip(1),
			transforms.ToTensor(),
		])
		audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])

		#waveform, sr = torchaudio.load(wav_file)
		#subseq_len = waveform.shape[1] / num_subseqs
		spectrograms = []
		max_spec_shape = []
		for wave in wav_file:
			#for i in range(int(num_subseqs)):
			try:
				audio, sr = torchaudio.load(wave) #,
								#num_frames=int(subseq_len),
								#frame_offset=int(subseq_len*i))
			except:
				audio, sr = torchaudio.load(wave) #,
			#x = vggish_input.waveform_to_examples(audio.numpy(), self.sample_rate)
			if audio.shape[1] <= 45599:
				_audio = torch.zeros((1, 45599))
				_audio[:, -audio.shape[1]:] = audio
				audio = _audio
			audiofeatures = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=882, hop_length=441, n_mels=64,
												   n_fft=1024, window_fn=torch.hann_window)(audio)
			#waveform, sr = torchaudio.load(audioPath, frame_offset=int(subseq_len*i), num_frames=int(subseq_len))

			max_spec_shape.append(audiofeatures.shape[2])

			#if audiofeatures.shape[2]> 400:
			#	print(wave)

			#if specgram.shape[2] > 851:
			#	_audio_features = torch.zeros(1, 64, 851)
			#	_audio_features = specgram[:, :, -851:]
			#	audiofeatures = _audio_features
			#elif specgram.shape[2] < 851:
			#	_audio_features = torch.zeros(1, 64, 851)
			#	_audio_features[:, :, -specgram.shape[2]:] = specgram
			#	audiofeatures = _audio_features
			#else:
			#	audiofeatures = specgram
			#if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
			#	waveform, sr = torchaudio.load(audioPath)
			#else:
			#	waveform, sr = torchaudio.load(audioPath)
			#specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, win_length=400, hop_length=160, n_mels=128, n_fft=1024, normalized=True)(audio)
			#specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, win_length=640, hop_length=640, n_mels=128, n_fft=1024, normalized=True)(waveform)
			audio_feature = audio_spec_transform(audiofeatures)

			# my transform
			#tensor = specgram.numpy()
			#res = np.where(tensor == 0, 1E-19 , tensor)
			#spectre = torch.from_numpy(res)

			#mellog_spc = spectre.log2()[0,:,:]#.numpy()
			#mean = mellog_spc.mean()
			#std = mellog_spc.std()
			#spec_norm = (mellog_spc - mean) / (std + 1e-11)
			#spec_min, spec_max = spec_norm.min(), spec_norm.max()
			#spec_scaled = (spec_norm/spec_max)*2 - 1
			spectrograms.append(audio_feature)
		spec_dim = max(max_spec_shape)

		audio_features = torch.zeros(len(max_spec_shape), 1, 64, spec_dim)
		for batch_idx, spectrogram in enumerate(spectrograms):
			if spectrogram.shape[2] < spec_dim:
				#print(batch_idx)
				#_audio_features = torch.zeros(1, 64, spec_dim)
				audio_features[batch_idx, :, :, -spectrogram.shape[2]:] = spectrogram
				#_audio_features[:, :, -spectrogram.shape[2]:] = spectrogram
				#audiofeatures = _audio_features
			else:
				audio_features[batch_idx, :,:, :] = spectrogram
		#melspecs_scaled = torch.stack(audio_features)

		#torch.cuda.synchronize()
		#t12 = time.time()
		return audio_features # melspecs_scaled

