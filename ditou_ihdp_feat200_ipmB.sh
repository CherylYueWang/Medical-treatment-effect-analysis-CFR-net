#
# Requires download and extraction of IHDP_100. See README.md
#

mkdir results
mkdir results/ditou_ihdp_feat200_ipmB

#export CUDA_VISIBLE_DEVICES=0

# run the param search for 20 iterations.
python cfr_param_search.py configs/ditou_ihdp_feat200_ipmB.txt 100

python evaluate.py configs/ditou_ihdp_feat200_ipmB.txt 1
