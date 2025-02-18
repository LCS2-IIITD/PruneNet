# PruneNet: Calibration-Free Model Compression [ICLR 2025]

This repository contains the code for the paper
[You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning](https://arxiv.org/abs/2501.15296)
(ICLR'25).

The paper introduces PruneNet, a novel structured-pruning technique
which intrinsically prunes transformer models without relying on any calibration
datasets. PruneNet works by slicing-off the unimportant rows from the
weight matrices of FFN layers of these models, where the importance scores of
the rows are computed using a two-layered neural network. The pruning process is
modeled as a stochastic policy which is trained to preserve the spectral
structure of the weight matrices using a standard RL-based pipeline.

The main scripts are in the `experiments/` folder. Our code utilizes scripts
from the [SliceGPT](https://github.com/microsoft/TransformerCompression)
repository. Visit their repository to get installation instructions.

# Running PruneNet

As an example, to run PruneNet on `microsoft/phi-2` with a
compression ratio of $0.25$ with fine-tuning of the compressed model, do the
following from the `experiments` folder:

    CUDA_VISIBLE_DEVICES=1 python trainable_activation_sparsity.py
        --log DEBUG                                
        --use_gpu                                  
        --model_name microsoft/phi-2               
        --num_episodes 15                          
        --learning-rate-action 0.0001              
        --sparsity_level 0.25                      
        --ppl-eval-dataset wikitext2               
        --finetune-dataset wikitext2               
        --finetune-train-nsamples 8000             
        --finetune-train-seqlen 1024               
        --finetune-train-batch-size 3              
        --lora-alpha 10                            
        --lora-r 32                                
        --lora-dropout 0.05                        
        --lora-target-option attn_head_and_mlp     
        --eval-steps 16                            
        --save-steps 16                            
        --epochs 1                                 
        --model_save_path "../models/"             
        --sparsity_technique bernoulli

The weights of the trained action model (which computes the importance scores)
will be saved in the `../models/` directory. This action model can then be
reused to slice any LLM (see the script for more details).

The results reported in the paper (for models from the LLaMa, OPT, and Phi
series) were generated using models compressed with the
`experiments/run_all_slicing*` scripts. For a detailed example with
`microsoft/phi-2`, see `experiments/run_all_slicing_phi.sh`.

# Evaluation scripts

We re-use the LM evaluation scripts from
[SliceGPT](https://github.com/microsoft/TransformerCompression) to evaluate our
compressed models. See `experiments/run_lm_eval.py` for details. See the
`experiments/run_llm_eval*` scripts for details on how we evaluate the models.
For our running example of `microsoft/phi-2`, the script
`experiments/run_llm_eval_phi.sh` is helpful.

# Slicing the attention modules

In addition to slicing the FFN weight matrices, the scripts
`experiments/trainable_activation_sparsity_allmodules.py` and
`experiments/run_lm_eval_allmodules.py` slice the attention modules using the
same pruning technique. However, we observed that doing this harms the
compressed model's performance significantly, and this step is therefore not
advised.

# Citation

If you find our work useful in your projects/research, kindly cite our paper:

    @inproceedings{
        sengupta2025you,
        title={You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning},
        author={Ayan Sengupta and Siddhant Chaudhary and Tanmoy Chakraborty},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=5RZoYIT3u6}
    } 
