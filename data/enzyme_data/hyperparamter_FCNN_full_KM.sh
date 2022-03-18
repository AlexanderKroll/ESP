#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=5GB:accelerator_model=gtx1080ti
#PBS -A kcat-prediction
#PBS -N hyperparamter_FCNN_full_KM
#PBS -j oe
#PBS -o "hyperparamter_FCNN_full_KM.log"
#PBS -r y
#PBS -m ae
#PBS -M alkro105@hhu.de
#PBS -J 0-20

#cd $PBS_O_WORKDIR

module load Python/3.8.3
module load CUDA/10.1.243
module load cuDNN/7.6.5

python3 /home/alkro105/KM_paper/hyperparamter_FCNN_full_KM.py $PBS_ARRAY_INDEX