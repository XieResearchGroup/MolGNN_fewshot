#### GIN fine-tuning

device=0
experiment_date=0312_0
split=scaffold

input_model_file=trained_model/chemblFP_full_batch128

epochs=1
eval_train=0
### for GIN
for runseed in 0
do
#for dataset in bace
#for dataset in jak1 jak2 jak3
#for dataset in bace bbbp sider clintox toxcast 
#for dataset in  tox21  hiv muv
for dataset in amu
do

python finetune.py --input_model_file ${input_model_file}  --split ${split} --filename ${dataset}_FULL_finetune_${experiment_date} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} \
> outlogs0306/finetune_seed${runseed}_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"
done
done
