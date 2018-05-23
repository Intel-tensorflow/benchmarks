#!/bin/bash

export PYTHONPATH=/dataset/sfu2/nervana/install-numa-public
export NTHREAD=56
export OPENBLAS_NUM_THREADS=1

export OMP_NUM_THREADS=$NTHREAD


python  tf_cnn_benchmarks.py  --mkl=True --forward_only=False --num_batches=100 --kmp_blocktime=0 --num_warmup_batches=0 --num_inter_threads=2 --distortions=False --optimizer=sgd --batch_size=128  --num_intra_threads=$NTHREAD --data_format=NCHW --num_omp_threads=$NTHREAD --model=resnet50 --device numa --num_gpus 2 --gpu_indices 0,1 
