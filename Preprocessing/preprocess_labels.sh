#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --mem=12000M
#SBATCH --job-name=preprocess_annot
#SBATCH --time=15-00:00:00
#SBATCH --mail-user=gnana-praveen.rajasekhar@crim.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
export PATH="/misc/home/reco/rajasegp/anaconda3/bin:$PATH"
#source activate py36_ssd20
source activate Multimodal
scripts_dir=`pwd`
MatlabFE=`pwd`
mdl=senet18e17
lf=ocsoftmax
atype=LA
#config=config_image.txt

#ptd=/misc/archive07/patx/asvspoof2019
#ptf=/lu/fast_scratch/patx/alamja/anti-spoofing-10dec2020/${atype}pickle
#ptp=/misc/archive07/patx/asvspoof2019/${atype}/ASVspoof2019_${atype}_protocols
#mdl_dir=/lu/fast_scratch/patx/alamja/anti-spoofing-10dec2020/models/$atype/$mdl/$lf
##mdl_dir=/lu/fast_scratch/patx/alamja/anti-spoofing-10dec2020/models_ensem/$atype/$mdl/$lf
#mkdir -p ${mdl_dir}/
#echo "This is array task ${SLURM_ARRAY_TASK_ID}, the sample name is ${data_start_range} and the sex is ${data_end_range}." >> output.txt
#python train_senet18e17.py --add_loss ${lf} -o ${mdl_dir} --gpu 0 -a ${atype} -d ${ptd} -f ${ptf} -p ${ptp}
python3 preprocess_labels.py
wait
echo "DONE"
