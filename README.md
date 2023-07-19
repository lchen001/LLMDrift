# 🎓 LLM Drifts: How Is ChatGPT’s Behavior Changing over Time?


Large language models (LLM) services such as GPT-4 and GPT-3.5 are widely being used. However, when and how these models are updated over time is opaque. Towards filling in this gap, this repository contains (i) a diverse set of _datasets_, and (ii) _generations_ from popular LLMs (including GPT-4 and GPT-3.5) on these datasets over time. 



## 🔍 Main Findings

<p align="center">
  <img align="center" src="asset/intro.png" width="460px" />
</p>
<p align="left">
  <b>Figure 1:</b> Performance of the March 2023 and June 2023 versions of GPT-4 and GPT-3.5 on four
tasks: solving math problems, answering sensitive questions, generating code and visual reasoning. The
performances of GPT-4 and GPT-3.5 can vary substantially over time, and for the worse in some tasks.
</p>


What are the main findings? In a nutshell, there are many interesting performance shifts over time. For example, GPT-4 (March 2023) was very good at identifying prime numbers (accuracy 97.6%)
but GPT-4 (June 2023) was very poor on these same questions (accuracy 2.4%). Interestingly
GPT-3.5 (June 2023) was much better than GPT-3.5 (March 2023) in this task. We hope releasing the datasets and generations can help the community to understand how LLM services drift better. The above figure gives a quantatitive summary.

## 🚀 Reproducing the Results

You can directly run the  [Google Colab Notebook](https://colab.research.google.com/github/lchen001/LLMDrift/blob/main/Intro.ipynb) to reproduce the monitored performance drifts in [our paper](https://arxiv.org/pdf/2307.09009.pdf). You don't need API keys to get started. You can also use the [local intro notebook](intro.ipynb) directly.


## 💾 Datasets and Generations
The datasets and generations can be found under ```generation/```. Each csv file corresponds to one dataset. One record/row corresponds to one query and the generation from one LLM service.


<p align="center">
  <img align="center" src="asset/generationexample.png" width="460px" />
</p>
<p align="left">
  <b>Figure 1:</b> The first few rows in the LLM  generations on PRIME dataset.
</p>

The above figure shows the first few rows in the ```generation/PRIME_EVAL.csv```. It includes the model, query parameters (such as temperature), the query, the reference answer, the generated answer, and latency.  Such information could be leverage to study various aspects of LLM services.

## 📚 Read More


You can find more details in the academic paper:
* [**How Is ChatGPT’s Behavior Changing over Time?**](https://arxiv.org/pdf/2307.09009.pdf)

## 🎯 Reference

If you use our findings and/or datasets in a research paper, please cite our work as follows:


```
@article{chen2023LLMDrift,
  title={How Is ChatGPT’s Behavior Changing over Time?},
  author={Chen, Lingjiao and Zaharia, Matei and Zou, James},
  journal={arXiv preprint arXiv:2307.09009},
  year={2023}
}
```