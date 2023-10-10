import sys
import os
import numpy as np
#from moviepy.editor import VideoFileClip
import scipy.io.wavfile
import datetime
import os
from tqdm import tqdm
import subprocess
from sox import Transformer
import argparse

parser = argparse.ArgumentParser(description='Create Configuration')
parser.add_argument('--data_start_range', type=str, help='optional filename', 
       default="")
parser.add_argument('--data_end_range', type=str, help='optional filename', 
       default="")

args = parser.parse_args()

video_files_dir = '/misc/lu/fast_scratch/patx/rajasegp/Affwild2/Data/Video_files'
output_dir = '/misc/lu/fast_scratch/patx/rajasegp/Affwild2/Data/SegmendtedAudioFiles/Shift_1_win_32'
timestamps_dir = '/misc/lu/bf_scratch/patx/rajasegp/RecurrentJointAttentionwithLSTMs/datasets/realtimestamps'
audio_dir = '/misc/lu/fast_scratch/patx/rajasegp/Affwild2/Data/Audio_files'
temp_dir = '/misc/lu/fast_scratch/patx/rajasegp/Affwild2/Data/temp_files'

def extract_audio(video):
	if not os.path.isdir(audio_dir):
		os.makedirs(audio_dir)
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	if not os.path.isdir(timestamps_dir):
		os.makedirs(timestamps_dir)
	if not os.path.isdir(os.path.join(temp_dir,os.path.splitext(video)[0])):
		os.makedirs(os.path.join(temp_dir,os.path.splitext(video)[0]))
	mkvfile = os.path.join(temp_dir,os.path.splitext(video)[0], 'temp.mkv')
	command = 'mkvmerge -o ' + mkvfile + ' ' + os.path.join(video_files_dir,video)
	subprocess.call(command, shell=True)
	video_ts_file = os.path.join(timestamps_dir, os.path.splitext(video)[0] + '_video_ts.txt')
	audio_ts_file = os.path.join(temp_dir,os.path.splitext(video)[0], 'audio_ts.txt')
	command = 'mkvextract ' + mkvfile + ' timestamps_v2 0:' + video_ts_file
	subprocess.call(command, shell=True)
	command = 'mkvextract ' + mkvfile + ' timestamps_v2 1:' + audio_ts_file
	subprocess.call(command, shell=True)
	with open(video_ts_file, 'r') as f:
		f.readline()  # skip header
		video_start = f.readline()
	with open(audio_ts_file, 'r') as f:
		f.readline()  # skip header
		audio_start = f.readline()
	offset_ms = int(audio_start) - int(video_start)
	# extract audio
	audio_tmp = os.path.join(temp_dir,os.path.splitext(video)[0], 'temp.wav')
	command = 'ffmpeg -i ' + os.path.join(video_files_dir, video) + ' -ar 44100 -ac 1 -y ' + audio_tmp
	subprocess.call(command, shell=True)
	# use the offset to pad the audio with zeros, or trim the audio
	audio_name = os.path.join(audio_dir,os.path.splitext(video)[0] + '.wav')
	tfm = Transformer()
	if offset_ms >= 0:
		tfm.pad(start_duration=offset_ms / 1000)
	elif offset_ms < 0:
		tfm.trim(start_time=-offset_ms / 1000)
	tfm.build(audio_tmp, audio_name)
	os.remove(mkvfile)
	os.remove(audio_tmp)
	#os.remove(video_ts_file)
	os.remove(audio_ts_file)
	return audio_name

def main():
	start_range = args.data_start_range
	end_range = args.data_end_range
	video_files = os.listdir(video_files_dir)[int(start_range):int(end_range)]
	#video_files = os.listdir(video_files_dir)[0:10]

	for video in tqdm(video_files):
		#video = "119.avi"
		audio_file_name = extract_audio(video)
		file_name = os.path.splitext(audio_file_name)[0]
		file_name = os.path.split(file_name)[1]
		out_file_dir = os.path.join(output_dir, file_name)
		if not os.path.isdir(out_file_dir):
			os.makedirs(out_file_dir)

		time_filename = os.path.join(timestamps_dir, file_name) + '_video_ts.txt'
		#if not os.path.exists(time_filename):
		#	continue
		f = open(os.path.join(time_filename))
		lines = f.readlines()[1:]
		num_files = len(lines)

		for j in range(num_files):
			if j<31:
				start_time = 0.0 #float(lines[j]) / 1000
			else:
				start_time = float(lines[j-31]) /1000
			end_time = float(lines[j]) /1000
			if end_time == 0.0:
				end_time = float(lines[j+1]) /1000
			st_time = str(datetime.timedelta(seconds=start_time))
			en_time = str(datetime.timedelta(seconds=end_time))
			output_file_name = os.path.join(out_file_dir, str(j+1)) + '.wav'
			mycmd = "ffmpeg -i " + audio_file_name + ' -ss '+ st_time + ' -to ' + en_time +' -ar 44100 -q:a 0 -map a '+ output_file_name
			os.system(mycmd)

if __name__ == "__main__":
	main()
