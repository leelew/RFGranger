# RFGranger | Nonlinear Granger causality test based on random forest

[Lu Li](https://scholar.google.com.hk/citations?user=2eKsP0gAAAAJ&hl=zh-CN), 

### Introduction

This repository contains the models described in the paper "A causal inference model based on Random Forest to identify soil moisture-precipitation feedback". We developed a causal-inference model based on the **Granger causality analysis** and **a nonlinear machine learning model**. This model includes three steps: **nonlinear anomaly decomposition, nonlinear Granger causality analysis, and evaluation of the quality of SM-P feedback**, which eliminates the nonlinear response of **interannual and seasonal variability, the memory effects of climatic factors** and isolates the causal relationship of local SM-P feedback.

### Edition

RFGranger is implemented by two programme language, [MATLAB]() and [python](). Two editions use the same function and file names. MATLAB edition now has some bugs needs to improved. 

### Citation

If you use these models in your research, please cite:

```bibtex
@article{Lu Li,
	author = {Lu Li, Wei Shangguan, Yongjiu Dai et al.},
	title = {A causal inference model based on Random Forest to identify soil moisture-precipitation feedback},
	journal = {Journal of Hydrometeorlogy},
	year = {2020}
}
```

### [License](https://github.com/leelew/NGCF/blob/master/LICENSE)

Copyright (c) 2019, Lu Li
