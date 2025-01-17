
## Universal Multi-view Consensus Graph Learning (UMCGL)

### Introduction

This is an implement of methods with PyTorch in "UMCGL: Universal Multi-view Consensus Graph Learning with Consistency and Diversity" that published in *IEEE Transactions on Image Processing*.
UMCGL is run on a server with a standard Ubuntu-16.04 operation system with a NVIDIA Tesla P100 16G GPU and 126G RAM.
### Datasets Descriptions

- Original multi-view datasets are stored in **/_multiview datasets**.
- Original graph matrices are stored in **/datasetW**, which are generated by KNN (see Matlab codes in **/datasetW/ConstructW** files).
- For all datasets, please obtain them from the following links: <https://drive.google.com/drive/folders/1Po0Im25dogk4scK0Sm_FqaGg1P93F4Hd?usp=drive_link>;
<https://pan.baidu.com/s/1bUUz3lEWTi1V8pEemTh0Aw?pwd=59e0> (Extracted code：59e0).

### Requirements

Require Python 3.7.2

- torch 1.7.1
- numpy 1.21.6
- scikit-learn 1.0.2

### Quick Running and Testing

- Run  `run_example.py` for multi-view complete and incomplete clustering tasks to see the example performance. 
- If you want to test your own datesets, you can put the datasets into **/_multiview datasets** and obtain their graphs by ConstructW.zip. Then, you can move them into **/datasetW** and train the corresponding models by employing these graphs. Finally, you can set the configuration file according to the example **/config**.

### Reference
```
@ARTICLE{Du2024UMCGL,
  author={Du, Shide and Cai, Zhiling and Wu, Zhihao and Pi, Yueyang and Wang, Shiping},
  journal={IEEE Transactions on Image Processing}, 
  title={{UMCGL}: Universal Multi-view Consensus Graph Learning with Consistency and Diversity}, 
  year={2024},
  volume={33},
  pages={3399-3412},
}
```

If you have any questions, please feel free to contact dushidems@gmail.com at any time. Thanks.