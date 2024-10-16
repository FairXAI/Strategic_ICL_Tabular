# Strategic Demonstration Selection for Improved Fairness in LLM In-Context Learning

[![arXiv](https://img.shields.io/badge/arXiv-2408.09757-b31b1b.svg)](https://www.arxiv.org/abs/2408.09757)

Demo page: https://fairxai.github.io/demo/TabularICL/index.html

**Package Requirement**: openai, numpy==1.25, sklearn, fairlearn, tenacity

**Abstract:** Recent studies highlight the effectiveness of using in-context learning (ICL) to steer large language models (LLMs) in processing tabular data, a challenging task given the structured nature of such data. Despite advancements in performance, the fairness implications of these methods are less understood. This study investigates how varying demonstrations within ICL prompts influence the fairness outcomes of LLMs. 

Our findings reveal that deliberately including minority group samples in prompts significantly boosts fairness without sacrificing predictive accuracy. Further experiments demonstrate that the proportion of minority to majority samples in demonstrations affects the trade-off between fairness and prediction accuracy. 

Based on these insights, we introduce a mitigation technique that employs clustering and evolutionary strategies to curate a diverse and representative sample set from the training data. This approach aims to enhance both predictive performance and fairness in ICL applications. Experimental results validate that our proposed method dramatically improves fairness across various metrics, showing its efficacy in real-world scenarios.

-----------------------------------------------


#### Quick Running

Default settings on Adult dataset, gpt-3.5-turbo, 16 demonstrations, S1 strategy.

``
python main.py 
``

#### Customised Running

Define customised parameters. 

``
python main.py --dataset [Cora/Credit] --k
`` [16/8/4]  ...

Other possible options of parameters are as below.

```python
#LLMs
llm=[gpt-4-1106-preview,gpt-3.5-turbo,text-davinci-003, etc]

#Credit card clients Dataset (Credit, Yeh (2016)) and (2) adult income (Adult, Becker & Kohavi (1996))
dataset=[Adult,Credit]

#Selection strategies: (1) S1: Balanced Samples with Balanced Labels; (2) S2: Prioritize Minority Samples with Balanced Labels; (3) S3: Prioritize Minority Samples with Unbalanced Labels. #todo clean the options to s1/s2/s3
adult_strategy set = [Balanced_FM_LH, All_F_Balanced_LH, All_F_All_L, All_F_All_H, All_M_All_H]
credit_strategy set= [Balanced_FM_OntimDue,All_F_Balanced_OntimDue,All_F_All_Ontime,All_F_All_Overdue,All_M_All_Overdue]
```


-----------------------------------------------
## Citation

```
@article{hu2024strategic,
  title={Strategic Demonstration Selection for Improved Fairness in LLM In-Context Learning},
  author={Hu, Jingyu and Liu, Weiru and Du, Mengnan},
  journal={arXiv preprint arXiv:2408.09757},
  year={2024}
}
```

