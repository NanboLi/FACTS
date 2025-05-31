## Multivariate Time-Series Forecasting (MTSF):  
In [`time_series/modules.py`](./modules.py), we provide modular components that allow easy customisation of FACTS for *(Multivariate) Time-Series Forecasting* tasks. We also provide **FACTS_MTS** model for general MTS tasks in [`time_series/FACTS_MTS.py`](./FACTS_MTS.py), which was used for the MTS experiments in our paper. Note that, originally, we made **FACTS_MTS** a plug-and-use with the [TSLib Benchmark](https://github.com/thuml/Time-Series-Library). However, due to swift version changes of the TSLib, our **FACTS_MTS** code might not be directly compatible anymore. We will try to close the gap in a future release.


### How to use in general:
1. Ensure our **facts_ssm** package is installed (see the [installation instructions](../../../README.md))
2. Simply import the **FACTS_MTS** module:\
   ```from facts_ssm.demos.time_series import FACTS_MTS```\
   and configure it.


### How to use FACTS_MTS with the TSLib
(TODO...)