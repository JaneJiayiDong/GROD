set -e

for d in mESC mHSC-E mHSC-GM mDC hESC hHep
do
for seed in 50 51 52 53 54 55 56 57 58 59
do
    python main.py \
    -seed $seed \
    -data /$d/1000_$d/ \
    -n_nodes 1000 \
    -k 0.1 \
    -s 1
done
done