#!/bin/bash
#PBS -N shortterm_infer          
#PBS -q GPU-LA                   
#PBS -j oe                      
#PBS -o logs/infer_$PBS_JOBID.log  

cd $PBS_O_WORKDIR                
module load cuda/12.1            
source activate MMT         

python inference.py
