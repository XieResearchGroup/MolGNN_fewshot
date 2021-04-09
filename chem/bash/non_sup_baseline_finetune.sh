#### benchmark fine-tuning

device=2
experiment_date=0409


split=random_scaffold


input_model_file=/raid/home/public/YangLiu/contextPred/pretrained_models/zincStandard_pretrained_model_with_contextPred
emb_dim=300
epochs=100
eval_train=1
gnn_type=gin
use_original=1

dropout_ratio=0.5
dataset=amu
seed=15
for runseed in  5 6 7 8
do



python finetune.py --input_model_file ${input_model_file}  --split ${split} --filename ${dataset}_NOSUP_BASE_${dropout_ratio}_splitseed_${seed}_${split}_finetune_${experiment_date} --dropout_ratio ${dropout_ratio}  --seed ${seed} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} --emb_dim ${emb_dim} --gnn_type ${gnn_type}  --use_original ${use_original} \
> base_corrected/NOSUP_Base_finetune_seed${runseed}_${dropout_ratio}_splitseed_${seed}_${split}_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"

done 