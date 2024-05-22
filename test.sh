#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=16G

source ../cuda11.8.sh

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ssnet_new
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.8/site-packages/tensorrt_libs
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/miyachi/cuda-11.8
export TF_XLA_FLAGS=--tf_xla_enable_xla_devices

export PATH=/home/miyachi/julia-1.10.3/bin:$PATH
export JULIA_PROJECT="."
julia basic.jl