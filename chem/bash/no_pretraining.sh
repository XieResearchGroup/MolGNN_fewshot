#### GIN fine-tuning

device=3
experiment_date=0405
split=random_scaffold



epochs=100
eval_train=1
#dropout_ratio=0.7

dropout_ratio=0.5
dataset=jak3
seed=66

for runseed in 0 1 2 3 4
do
#for dataset in jak2 jak3

#for dataset in bace
#for dataset in jak1 jak2 jak3
#do
#for dataset in ellinger mpro
#do
#for dataset in bace bbbp sider clintox toxcast 
#for dataset in   muv
#for dataset in amu ellinger mpro jak1 jak2 jak3
#for seed in 27 16 0 1 6 7 
#for seed in 15 19 21 24 26 28

python finetune.py  --split ${split} \
--filename ${dataset}_${dropout_ratio}_splitseed_${seed}_${split}_NOPRE_finetune_${experiment_date} --dropout_ratio ${dropout_ratio}  --seed ${seed} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} \
> ours_more/finetune_runseed${runseed}_splitseed_${seed}_${split}_NOPRE_${dropout_ratio}_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"
done





