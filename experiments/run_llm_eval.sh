for model in facebook/opt-125m facebook/opt-1.3b facebook/opt-2.7b #facebook/opt-6.7b
do

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0  python run_lm_eval.py  \
    --model ${model}    \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\
    --model-save-path '../models/'

for dataset in wikitext2 #alpaca
do

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0  python run_lm_eval.py  \
    --model ${model}    \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\
    --finetune \
    --finetune-dataset ${dataset} \
    --model-save-path '../models/'

done
done
