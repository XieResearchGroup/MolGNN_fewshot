device=0
experiment_date=0407


split=random_scaffold


input_model_file=supervised_contextpred.pth

epochs=100
eval_train=1
gnn_type=gin

seed=66

dataset=jak3
for runseed in 1 2 3 4


do
nohup time
python finetune.py --input_model_file ${input_model_file}  --split ${split} --filename ${dataset}_2F_BASE_${dropout_ratio}_splitseed_${seed}_${split}_finetune_${experiment_date}   --seed ${seed} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train}  --gnn_type ${gnn_type}  \
> base_corrected/2F_Base_finetune_seed${runseed}_splitseed_${seed}_${split}_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"
done