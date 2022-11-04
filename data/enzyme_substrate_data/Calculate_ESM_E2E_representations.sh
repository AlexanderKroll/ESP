#!/bin/bash
#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=40GB:accelerator_model=rtx6000
#PBS -A kcat-prediction
#PBS -N extract_E2E
#PBS -j oe
#PBS -o "extract_E2E_GO.log"
#PBS -r y
#PBS -m ae
#PBS -M alkro105@hhu.de


#cd /home/alkro105/enzym_rep/ESM_binary

module load Python/3.6.5
module load CUDA/11.0.2
module load cuDNN/8.0.4

python3 /home/alkro105/enzym_rep/ESM_binary/extract_TSP.py esm1b_t33_650M_UR50S /gpfs/project/alkro105/all_transporter_sequences_TSP.fasta /gpfs/project/alkro105/all_transporter_sequences_TSP_esm1b_ts --repr_layers 33 --include mean
python3 /home/alkro105/enzym_rep/ESM_binary/merge_pt_files.py "all_transporter_sequences_TSP_esm1b_ts"
