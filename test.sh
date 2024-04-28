#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=16G

module use /usr/local/package/modulefiles
module load julia

export JULIA_PROJECT="."
julia basic.jl