
#### FP pretraining 
experiment_date=0311
device=0


input_file=trained_model/ContextPred_ZINC_onehotMLP

output_model=trained_model/chemblFP_full_batch128

epochs=100

dataset=/raid/home/public/dataset_ContextPred_0219/ChemBL
batch_size=128

nohup time  
python pretrain_fingerprint.py --dataset  ${dataset}  --input_model_file ${input_file} \
--output_model_file ${output_model}  --batch_size ${batch_size} \
--epochs ${epochs} --device ${device}  > outlogs0306/pre_chemblFP_FULL_our${experiment_date}_output.log 2>&1 &
echo "done chemblFPFULL_${experiment_date}"
