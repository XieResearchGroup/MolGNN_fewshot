#### GIN fine-tuning

device=0
experiment_date=0405
split=random_scaffold


input_model_file=trained_model/chemblFP_full_batch128

epochs=100
eval_train=1
#dropout_ratio=0.7

dropout_ratio=0.5

dataset=amu
if [$dataset==amu]
then
  seed=15
elif [$dataset==ellinger]
then
  seed=4
elif [$dataset==mpro]
then
  seed=7
elif [$dataset==jak1]
then
  seed=20
elif [$dataset==jak2]
then
  seed=4
elif [$dataset==jak3]
then
  seed=66
fi

for runseed in 0 1 3 4
do


python finetune.py --input_model_file ${input_model_file}  --split ${split} \
--filename ${dataset}_${dropout_ratio}_splitseed_${seed}_${split}_FULLchembl_finetune_${experiment_date} --dropout_ratio ${dropout_ratio}  --seed ${seed} \
--device $device --runseed $runseed  --dataset ${dataset} --epochs ${epochs} --eval_train ${eval_train} \
> ours_more/finetune_runseed${runseed}_splitseed_${seed}_${split}_FULLchembl_${dropout_ratio}_${experiment_date}_${dataset}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"

done





