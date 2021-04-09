
#### FP pretraining 
experiment_date=0222_0
device=2

 


output_model=trained_model/ContextPred_ZINC_onehotMLP

epochs=100


dataset=dataset/zinc_standard_agent

node_feat_dim=154
emb_dim=256


nohup time  
python pretrain_contextpred.py --dataset  ${dataset}  --node_feat_dim  ${node_feat_dim} --emb_dim ${emb_dim} \
--output_model_file ${output_model} \
--epochs ${epochs} --device ${device}  > outlogs0220/ContextPred_ZINC_${experiment_date}_output.log 2>&1 &
echo "done ZINC_${experiment_date}"

