#!/bin/bash

#SBATCH --partition=GPU-a40       # select a partition i.e. "GPU-a100"
#SBATCH --gres=gpu:1              # Use GPU
#SBATCH --nodes=1                 # select number of nodes
#SBATCH --ntasks-per-node=2       # select number of tasks per node
######SBATCH --mem=32GB                  
#SBATCH --time=2-00:00:00         # Maximum time  
#SBATCH --exclude=a-a40-o-1

source /home/johannes.karwounopoulos/miniconda3/etc/profile.d/conda.sh
conda activate chemtorch

# The first three experiments use the ground truth coordinates 
# for training and validation. They only differ in the data source used for testing. 
# The last experiment is without coordinates.
# For each experiment we vary the data source, so we do each experiment first for the 
# random split (data/rgd1),then for the reaction core split (data/rgd1_rxn_split) and
# finally for the barrier split (data/rgd1_barrier_split).

# Ground truth coordinates for the test set
echo "Using ground truth coordinates for rgd1"
python scripts/main.py --multirun \
    +experiment=3d_features \
    data_ingestor=rgd1 \
    data_ingestor.data_source.data_folder=data/rgd1,data/rgd1_rxn_split,data/rgd1_barrier_split \
    seed=0,1,2 \
    log=true \
    routine.optimizer.lr=0.0001 \
    model.hidden_channels=900 \
    model.layer_stack.dmpnn_blocks.depth=3 \
    model.head.dropout=0 \
    model.head.hidden_size=100 \
    model.head.num_hidden_layers=0 \
    model.feature_hidden_channels=200 \
    model.feature_out_channels=256 > ground_truth_rgd1.log

# Diffusion coordinates for the test set we only use the random split
echo "Using diffusion coordinates for rgd1"
python scripts/main.py --multirun \
    +experiment=3d_features \
    data_ingestor=rgd1 \
    data_ingestor.data_source.data_folder=data/rgd1 \
    seed=0,1,2 \
    log=true \
    group_name=rgd1_diffusion \
    routine.optimizer.lr=0.001 \
    model.hidden_channels=900 \
    model.layer_stack.dmpnn_blocks.depth=2 \
    model.head.dropout=0 \
    model.head.hidden_size=300 \
    model.head.num_hidden_layers=0 \
    model.feature_hidden_channels=128 \
    model.feature_out_channels=256 \
    data_ingestor.data_source.test_coordinate=test_mace_mp_ts_diffusion.npz > diffusion.log


# Flow matching coordinates for the test set
echo "Using flow matching coordinates for rgd1"
python scripts/main.py --multirun \
    +experiment=3d_features \
    data_ingestor=rgd1 \
    data_ingestor.data_source.data_folder=data/rgd1,data/rgd1_rxn_split,data/rgd1_barrier_split \
    log=true \
    group_name=rgd1_flowMatching \
    seed=0,1,2 \
    routine.optimizer.lr=0.001 \
    model.hidden_channels=900 \
    model.layer_stack.dmpnn_blocks.depth=2 \
    model.head.dropout=0 \
    model.head.hidden_size=300 \
    model.head.num_hidden_layers=0 \
    model.feature_hidden_channels=128 \
    model.feature_out_channels=512 \
    model.features_dropout=0.1 \
    model.encoder.modified_in_channels=622 \
    data_ingestor.data_source.test_coordinate=test_mace_mp_ts_flowMatching.npz > flowMatching.log


# Using no coordinates
echo "Using no coordinates for rgd1"
HYDRA_FULL_ERROR=1 python scripts/main.py --multirun \
    +experiment=no_coordinates \
    data_ingestor=rgd1 \
    data_ingestor.data_source.data_folder=data/rgd1,data/rgd1_rxn_split,data/rgd1_barrier_split \
    log=true \
    group_name=rgd1_no_coordinates \
    seed=0,1,2 \
    routine.optimizer.lr=0.001 \
    model.head.dropout=0.1 \
    model.hidden_channels=600 \
    model.head.hidden_size=300 \
    model.layer_stack.dmpnn_blocks.depth=6 \
    model.head.num_hidden_layers=0 > no_coordinates_rgd1.log
