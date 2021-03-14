#### GIN fine-tuning

device=0
experiment_date=0313_train
split=random

input_model_file=trained_model/chemblFP_full_batch128

epochs=100
eval_train=1
dropout_ratio=0.9
### for GIN
for runseed in 0 2 
do
#for dataset in bace
#for dataset in jak1 jak2 jak3
#for dataset in amu ellinger mpro
#for dataset in bace bbbp sider clintox toxcast 
#for dataset in   muv
#for dataset in amu ellinger mpro jak1 jak2 jak3
for dataset in amu 
do

python finetune.py --input_model_file ${input_model_file}  --split ${split} \
--filename ${dataset}_FULL_batch128_train_finetune_${experiment_date} --dropout_ratio ${dropout_ratio} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} \
> outlogs0313/finetune_seed${runseed}_b128_train${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"
done
done
