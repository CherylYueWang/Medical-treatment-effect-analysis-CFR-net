#
# Requires download and extraction of IHDP_100. See README.md
#

mkdir results
mkdir results/cfr_ihdp_feat50

#export CUDA_VISIBLE_DEVICES=0

# run the param search for 20 iterations.
python cfr_param_search.py configs/cfr_ihdp_feat50.txt 100

python evaluate.py configs/cfr_ihdp_feat50.txt 1
