
#### FP pretraining 
experiment_date=0224_0
device=0


input_file=trained_model/ContextPred_ZINC_onehotMLP

output_model=trained_model/chemblFP_onehotMLP

epochs=100

dataset=/raid/home/public/dataset_ContextPred_0219/ChemBL


nohup time  
python pretrain_fingerprint.py --dataset  ${dataset}  --input_model_file ${input_file} \
--output_model_file ${output_model} \
--epochs ${epochs} --device ${device}  > outlogs0220/pre_chemblFP_onehot_MLP${experiment_date}_output.log 2>&1 &
echo "done chemblFP_${experiment_date}"
