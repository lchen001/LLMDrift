# 🎓 LLM Monitor Suite


Here is the code for monitoring LLM performance. 



## ⌨️ Usage

To use this suite, perform the following steps:

- Download the package and navigate to the monitor directory

```
git clone https://github.com/lchen001/LLMDrift 
cd LLMDrift/src/monitor/
```

- Install the required packages.


(Optional but recommended) Create a conda environment.

```
conda create -n LLMDrift python=3.9
conda activate LLMDrift
```

Install all dependent libraries.
 

```
pip install -r requirements.txt
```



- Next, you need to set up OPENAI\_API\_KEY by

```
export OPENAI_API_KEY='YOUR API KEY' 
```



(3) Now you can execute the following command to evaluate the performance of GPT-4 and GPT-3.5 in March and June on the prime dataset.

python3 run_monitor.py --config=../../config/evaluate\_primefull.json