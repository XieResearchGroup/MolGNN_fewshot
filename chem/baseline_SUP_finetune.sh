#### benchmark fine-tuning

device=3

experiment_date=0313


split=random


input_model_file=/raid/home/public/YangLiu/contextPred/pretrained_models/chemblFiltered_and_supervise_pretrained_model_with_contextPred

emb_dim=300
epochs=100
eval_train=1
gnn_type=gin
use_original=1

### for GIN
for runseed in 1
do
for dataset in amu ellinger mpro
#for dataset in bace
#for dataset in jak1 jak2 jak3 amu ellinger mpro 
#for dataset in bace bbbp sider clintox toxcast 
#for dataset in  tox21  hiv muv
#for dataset in bbbp
do

python finetune.py --input_model_file ${input_model_file}  --split ${split} --filename ${dataset}_BASE_train_finetune_${experiment_date} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} --emb_dim ${emb_dim} --gnn_type ${gnn_type}  --use_original ${use_original} \
> base_logs/Base_finetune_train_seed${runseed}_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"
done
done
