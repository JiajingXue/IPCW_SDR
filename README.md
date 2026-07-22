# IPCW_SDR
Sufficient dimension reduction for right-censored survival data

# Maintainer

Jiajing Xue,  [xuejiajing@stu.edu.xmu.cn](xuejiajing@stu.edu.xmu.cn)  

# Files and functions

* `Proposed.py`  
  This file contains the `torch.nn` class for the proposed DNN-based Semiparametric AFT model  
  Main class:
  * `CensoringCoxEstimator`  
    Estimating $G_X(t)$
  * `ScalarNet`
    Training each direction
  * `CoxNN`
    Downstream prediction using NN
    

* `Data_GP.py`  
  This file contains the function for similation data generation  
  Main function:
  * `generate_survival_data`  
    The function used to generate simulation data

* `functions.py`  
  This file contains functions for conducting similation for the proposed method.  
  
* `demo.py`  
  This provides an example.
  
