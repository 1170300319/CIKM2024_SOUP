# SOUP: A Unified Shopping Query Suggestion Framework to Optimize Language Model with User Preference
Code and data for SOUP: A Unified Shopping Query Suggestion Framework to Optimize Language Model with User Preference.

We release part of the code of our proposed method. The complete version will be released in the future.

## Requirements:
- Python 3.9.7
- PyTorch 1.10.1
- transformers 4.2.1
- tqdm
- numpy
- sentencepiece
- pyyaml

## Usage
Run training scripts in run_script

    ```
    bash scripts/pretrain_base_u2q.sh 4
    ```
Here *4* means using 4 GPUs to conduct parallel pretraining.

