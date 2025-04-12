# PruneNet: Calibration-Free Model Compression

This repository contains the code for the paper
[You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning](https://arxiv.org/abs/2501.15296).

The paper introduces PruneNet, a novel structured-pruning technique which
intrinsically prunes transformer models without relying on any calibration
datasets. PruneNet works by slicing-off the unimportant rows from the weight
matrices of FFN layers of these models, where the importance scores of the rows
are computed using a two-layered neural network. The pruning process is modeled
as a stochastic policy which is trained to preserve the spectral structure of
the weight matrices using a standard RL-based pipeline.

# Installation and requirements

We re-use many components from the
[SliceGPT](https://github.com/microsoft/TransformerCompression) pipeline. For
this, we recommend a Python version `>=3.10`. To install the required
components, run the following:

    git clone https://github.com/microsoft/TransformerCompression
    cd TransformerCompression
    pip install -e .[experiment,finetune]
    pip install git+https://github.com/pnnl/DDKS

# Usage

The main scripts are in the `prunenet` directory. `prunenet/prunenet.py` is the
script which trains a `SparsityPredictor` (the policy model) used to compute
importance scores for rows of weight matrices, and the same script uses such a
policy model to prune an LLM. `prunenet/prunenet_utils.py` contains some utility
functions used throughout our implementation. `prunenet/SparsityPredictor.py`
contains the PyTorch definition of the policy model.

Here is an example which prunes the `facebook/opt-125m` model for a compression
ratio of `0.3`. Most users should be able to run this example locally.

    CUDA_VISIBLE_DEVICES=0 python3 -m prunenet                  \
        --model_name facebook/opt-125m                          \
        --compression_ratio 0.3                                 \
        --save_dir  /home/codetalker7/compressed_models/opt/    \
        --device cuda:0

This script will train the action model (if it doesn't already exist in the
directory specified by `--save_dir`), save the action model, prune the model and
save the weights of the pruned model. The trained action model can be re-used to
compress other models as well.

## Evaluation scripts

<!-- We re-use the LM evaluation scripts from -->
<!-- [SliceGPT](https://github.com/microsoft/TransformerCompression) to evaluate our -->
<!-- compressed models. See `experiments/run_lm_eval.py` for details. See the -->
<!-- `experiments/run_llm_eval*` scripts for details on how we evaluate the models. -->
<!-- For our running example of `microsoft/phi-2`, the script -->
<!-- `experiments/run_llm_eval_phi.sh` is helpful. -->

## Slicing the attention modules

<!-- In addition to slicing the FFN weight matrices, the scripts -->
<!-- `experiments/trainable_activation_sparsity_allmodules.py` and -->
<!-- `experiments/run_lm_eval_allmodules.py` slice the attention modules using the -->
<!-- same pruning technique. However, we observed that doing this harms the -->
<!-- compressed model's performance significantly, and this step is therefore not -->
<!-- advised. -->

## Citation

If you find our work useful in your projects/research, kindly cite our paper:

    @inproceedings{
        sengupta2025you,
        title={You Only Prune Once: Designing Calibration-Free Model Compression With Policy Learning},
        author={Ayan Sengupta and Siddhant Chaudhary and Tanmoy Chakraborty},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=5RZoYIT3u6}
    }
