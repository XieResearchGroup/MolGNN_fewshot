#### GIN fine-tuning
runseed=$1
device=$2
experiment_date=0207
split=scaffold

input_model_file=trained_model/FPpretrained_chembl

epochs=100
eval_train=1
### for GIN

#for dataset in bace bbbp sider 
#for dataset in clintox muv tox21 toxcast hiv
for dataset in muv
do

python finetune.py --input_model_file ${input_model_file}  --split ${split} --filename ${dataset}_finetune_${experiment_date} \
--device $device --runseed $runseed --gnn_type gin --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} \
> outlogs0129/finetune_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"
done

