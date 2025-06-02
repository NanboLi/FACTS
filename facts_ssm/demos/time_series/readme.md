## Multivariate Time-Series Forecasting (MTSF):  
In [`time_series/modules.py`](./modules.py), we provide modular components that allow easy customisation of FACTS for *(Multivariate) Time-Series Forecasting* tasks. A ready-to-use **FACTS_MTS** model for general MTS tasks is provided in [`time_series/FACTS_MTS.py`](./FACTS_MTS.py), which was used in our MTS experiments as presented in our paper. 

Originally, **FACTS_MTS** was designed to be plug-and-play with the [TSLib Benchmark](https://github.com/thuml/Time-Series-Library). However, due to recent and frequent version changes in TSLib, our **FACTS_MTS** implementation may no longer be fully compatible. We aim to address this in a future release.


### How to use in general:
1. Ensure our **facts_ssm** package is installed (see the [installation instructions](../../../README.md))
2. Simply import the **FACTS_MTS** module:\
   ```from facts_ssm.demos.time_series import FACTS_MTS```\
   and configure it.

We also recommended the users to play around and build their own MTS models using [`from facts_ssm import FACTS`](./modules.py) and the modules in [`time_series/modules.py`](./modules.py). 


### How to use FACTS_MTS with the TSLib
Perhaps *the easiest way* to create a FACTS_MTS model that can be directly loaded by the TSLib [```exp_basic.py```](https://github.com/thuml/Time-Series-Library/blob/main/exp/exp_basic.py) is to create a file, e.g. ```FACTS.py``` under ```Time-Series-Library/models/```and then copy &. paste the below code:
```
from facts_ssm.demos.time_series import FACTS_MTS


class Model(FACTS_MTS):
   def __init__(self, configs):
      super().__init__(configs)
```
. After this, one needs to configure ```Time-Series-Library/exp/exp_basic.py``` and ```Time-Series-Library/run.py``` accordingly to ensure compatiability. We will also try to provide a bridge as a walk-around in a future release.


(Updating...)