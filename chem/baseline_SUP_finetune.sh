#### benchmark fine-tuning

device=0
experiment_date=0402


split=random_scaffold


input_model_file=/raid/home/public/YangLiu/contextPred/pretrained_models/chemblFiltered_and_supervise_pretrained_model_with_contextPred
emb_dim=300
epochs=100
eval_train=1
gnn_type=gin
use_original=1
seed=25
dropout_ratio=0.5
dataset=amu

for runseed in  2 1 7
do 

#for dataset in bace
#for dataset in jak1 jak2 jak3 
#for dataset in bace bbbp sider clintox toxcast 
#for dataset in  tox21  hiv muv
#for dataset in bbbp
#do
nohup time
python finetune.py --input_model_file ${input_model_file}  --split ${split} --filename ${dataset}_SUP_BASE_${dropout_ratio}_splitseed_${seed}_${split}_finetune_${experiment_date} --dropout_ratio ${dropout_ratio}  --seed ${runseed} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} --emb_dim ${emb_dim} --gnn_type ${gnn_type}  --use_original ${use_original} \
> base0401/SUP_Base_finetune_seed${runseed}_${dropout_ratio}_splitseed_${seed}_${split}_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"

done 