
#### FP pretraining 
experiment_date=0203_0
device=1

#runseed=0 

#input_file=chemblFiltered_pretrained_model_with_contextPred
input_file=zincStandard_pretrained_model_with_contextPred

output_model=trained_model/FPpretrained_chembl

epochs=100

datapath=/workspace/new_DeepChem/

### for GIN

dataset=ChemBL


nohup time  
python pretrain_fingerprint.py --dataset  ${dataset} --datapath ${datapath} --input_model_file ${input_file} \
--output_model_file ${output_model} \
--epochs ${epochs} --device ${device}  > ./outlogs0129/FP_pre_${dataset}_${experiment_date}_output.log 2>&1 &
echo "done ${dataset}_${experiment_date}"
