## Multivariate Time-Series Forecasting (MTSF):  
In [`time_series/modules.py`](./modules.py), we provide modular components that allow easy customisation of FACTS for *(Multivariate) Time-Series Forecasting* tasks. A ready-to-use **FACTS_MTS** model for general MTS tasks is provided in [`time_series/FACTS_MTS.py`](./FACTS_MTS.py), which was used in our MTS experiments as presented in our paper. 

Originally, **FACTS_MTS** was designed to be plug-and-play with the [TSLib Benchmark](https://github.com/thuml/Time-Series-Library). However, due to recent and frequent version changes in TSLib, our **FACTS_MTS** implementation may no longer be fully compatible. We aim to address this in a future release.


### How to use in general:
1. Ensure our **facts_ssm** package is installed (see the [installation instructions](../../../README.md))
2. Simply import the **FACTS_MTS** module:\
   ```from facts_ssm.demos.time_series import FACTS_MTS```\
   and configure it.


### How to use FACTS_MTS with the TSLib
(TODO...)