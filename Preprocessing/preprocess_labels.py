import pandas as pd
import os
import sys
anno_path = "/misc/lu/fast_scratch/patx/rajasegp/Affwild2/Annotations"

def produce_multi_task_videos():
	train_videos = os.listdir(os.path.join(anno_path, "VA_Estimation_Challenge", "Train_Set"))
	val_videos = os.listdir(os.path.join(anno_path, "VA_Estimation_Challenge", "Validation_Set"))
	test_videos = os.listdir(os.path.join(anno_path, "VA_Estimation_Challenge", "TestSet"))

	train_videos = list(set(train_videos))
	val_videos = list(set(val_videos))
	return train_videos,val_videos

train_videos,val_videos = produce_multi_task_videos()

def get_names(id):
	name = ""
	if id>=0 and id<10:
		name = "0000" + str(id)
	elif id>=10 and id<100:
		name = "000" + str(id)
	elif id>=100 and id<1000:
		name = "00" + str(id)
	elif id>=1000 and id<10000:
		name = "0" + str(id)
	else:
		name = str(id)
	return name

def produce_va_labels_for_one_video(video_name, flag):
	label_dict = {}

	f = open(os.path.join(anno_path,"VA_Estimation_Challenge",flag,video_name))
	lines = f.readlines()[1:]
	for i in range(len(lines)):
		l = lines[i].strip().split(",")
		if l[0] == "-5" or l[1] == "-5":
			print(l)
			continue
		frame = get_names(i+1)
		#if os.path.exists(os.path.join(img_path,video_name,frame+".jpg")):
		n = video_name.split(".")[0]+"/"+frame+".jpg"
		if n not in label_dict.keys():
			label_dict[n] = [float(l[0]),float(l[1]), frame]
		else:
			label_dict[n][0] = float(l[0])
			label_dict[n][1] = float(l[1])
			label_dict[n][2] = float(frame)
	return label_dict


def produce_anno_csvs(videos, flag):
	save_path = "/misc/lu/fast_scratch/patx/rajasegp/Affwild2/Annotations/preprocessed_VA_annotations/" + flag
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	for video in videos:
		label_dict = produce_va_labels_for_one_video(video, flag)
		data = pd.DataFrame()
		imgs,V,A,frame_id = [],[],[],[]
		for k,v in label_dict.items():
			imgs.append(k)
			V.append(v[0])
			A.append(v[1])
			frame_id.append(v[2])
		data["img"],data["V"],data["A"], data["frame_id"] = imgs,V,A, frame_id
		data.to_csv(os.path.join(save_path,video.split(".")[0]+".csv"))

def produce_total_csvs(flag):
	path = "/export/livia/home/vision/pgan/Datasets/Affwild2/annotations/preprocessed_all_labeled_images/" + flag
	csv_data_list = os.listdir(path)
	total_data = pd.DataFrame()
	imgs,V,A = [],[],[]
	for csv in csv_data_list:
		print(csv)
		data = pd.read_csv(os.path.join(path,csv))
		imgs.extend(data["img"].to_list())
		V.extend(data["V"].to_list())
		A.extend(data["A"].to_list())
	print(len(imgs),len(A),len(V))
	total_data["img"],total_data["V"],total_data["A"] = imgs,V,A
	total_data.to_csv("ABAW2_multi_task_training.csv")

def produce_category_csvs():
	path = "/export/livia/home/vision/pgan/Datasets/Affwild2/annotations/preprocessed_AV/Train_Set"
	csv_data_list = os.listdir(path)
	VA_spec_data = []
	multi_data = []
	for csv in csv_data_list:
		print(csv)
		total_data = pd.read_csv(os.path.join(path,csv))
		imgs, V, A = total_data["img"],\
						  total_data["V"],total_data["A"]
		for i in range(len(imgs)):
			if V[i] != -1 and A[i] != -1:
				 multi_data.append(total_data.iloc[i, :])
		print(len(multi_data))
		print(len(VA_spec_data))
	multi_data = pd.DataFrame(multi_data)

	multi_data.to_csv("multi_train_data.csv")


#produce_training_data()
#produce_category_csvs()
#produce_total_csvs()
#### for train and val
#count = 0
#video_data = [[train_videos, 'Train_Set'], [val_videos, 'Validation_Set']]
#for videos in video_data:
#	produce_anno_csvs(videos[0], videos[1])
##    flag = ['Train_Set', 'Val_Set']
##produce_anno_csvs(val_videos, 'Val_Set')

anno_path = '/misc/lu/fast_scratch/patx/rajasegp/Affwild2/Annotations/VA_Estimation_Challenge/TestSet/names_of_videos_in_each_test_set/Valence_Arousal_Estimation_Challenge_test_set_release.txt'
label_dict = {}
test_annot_path = "/misc/lu/fast_scratch/patx/rajasegp/Affwild2/Annotations/preprocessed_VA_annotations/Test_Set/"

f = open(anno_path)
lines = f.readlines()

for i in range(len(lines)):
	print(lines[i])
	f_name = lines[i].replace("\n", "")
	if f_name.endswith('_left'):
		vid_name = f_name[:-5]
	elif f_name.endswith('_right'):
		vid_name = f_name[:-6]
	else:
		vid_name = f_name
	file_name = os.path.join('realtimestamps',  vid_name + '_video_ts.txt')
	
	f = open(os.path.join(file_name))
	video_lines = f.readlines()[1:]
	length = len(video_lines) 

	for j in range(length):
		frame = get_names(j+1)	

		n = os.path.join(f_name, frame + '.jpg')
		if n not in label_dict.keys():
			label_dict[n] = [frame]
		else:
			label_dict[n][0] = float(frame)

	data = pd.DataFrame()
	imgs,frame_id = [],[]
	for k,v in label_dict.items():
		imgs.append(k)
		frame_id.append(v[0])
	data["img"], data["frame_id"] = imgs, frame_id
	data.to_csv(os.path.join(test_annot_path, f_name + ".csv"))
