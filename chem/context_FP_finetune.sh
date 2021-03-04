#### GIN fine-tuning

device=0
experiment_date=0228_5
split=random

input_model_file=trained_model/chemblFP_onehotMLP

epochs=100
eval_train=0
### for GIN
for runseed in 7
do
#for dataset in bace
for dataset in jak1 jak2 jak3
#for dataset in bace bbbp sider clintox toxcast 
#for dataset in  tox21  hiv muv
#for dataset in bbbp
do

python finetune.py --input_model_file ${input_model_file}  --split ${split} --filename ${dataset}_finetune_${experiment_date} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} \
> outlogs0220/finetune_seed${runseed}_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"
done
done
