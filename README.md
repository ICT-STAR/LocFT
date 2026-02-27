# LocFT-BF

This repository hosts the code and data for the paper: **[Fine-tuning Done Right in Model Editing](https://arxiv.org/abs/2509.22072)**.

### Requirements:

- **Environment**: `requirements.txt` (Please use Python 3.9+ for this repository)

  ```shell
  pip install -r requirements.txt
  ```

- **Datasets**: The processed data from [ZsRE](https://github.com/nicola-decao/KnowledgeEditor), [COUNTERFACT](https://rome.baulab.info/), and [WikiBigEdit](https://github.com/ExplainableML/WikiBigEdit) used in our paper are organized and provided in the `./data/` directory.
- **Configurations**: We have provided default configurations for fine-tuning various LLMs in the `./hparams/` directory (e.g., `./hparams/qwen2.5-7b.yaml`). You can modify these configuration files to specify the model path, device ID, or hyperparameters as needed for fine-tuning.

### ⚠️ Important Note on the EOS Token

Unlike previous model editing codebases, **we explicitly append an end-of-sequence (EOS) token to all target answers.**

* **Why?** While previous model editing implementations omit the EOS token, typically causing irrelevant or incorrect generation after the correct answer, standard finetuning practices strictly require it (e.g., LlamaFactory and verl).
* **Impact:** This ensures a fair and rigorous comparison and aligning model editing with mainstream LLM training practices.

### Editing/Fine-tuning:

To run the individual editing/fine-tuning procedure, you only need to specify the path to the configuration file.

```shell
python fine-tune.py --config_path ./hparams/qwen2.5-7b.yaml
```

### Evaluation

**Editing Metric Evaluation**

To evaluate editing metrics reliability (editing success rate) and generalization (success rate for rephrased prompts), you can run:

```shell
CUDA_VISIBLE_DEVICES=0 python eval_edit_metric.py --data_path ./data/zsre/zsre_eval_3k.json --model_path ./saves/qwen --tp_size 1 --num_samples 3000
```

**General Task Evaluation**

To evaluate the general capabilities of edited LLMs, we recommend using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for comprehensive assessment. Additionally, we plan to support representative task evaluation based on vllm in the near future.

### Reproducing Main Results

To seamlessly reproduce the main experimental results from our paper (Table 3), which evaluate across three datasets and three LLMs, we provide an automated shell script.

```shell
chmod +x main_exp.sh
./main_exp.sh 2>&1 | tee main_exp_logs.txt
```

### Citation

If you have any questions, please feel free to open an issue or contact us. And if you find our work helpful, please cite our paper~

```bibtex

@inproceedings{
yang2026finetuning,
title={Fine-tuning Done Right in Model Editing},
author={Wanli Yang and Rui Tang and Hongyu Zang and Du Su and Qi Cao and Jingang Wang and Huawei Shen and Xueqi Cheng and Fei Sun},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=cfHuA5jsPt}
}

```
