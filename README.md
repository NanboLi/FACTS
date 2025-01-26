# FACTS
PyTorch Implementation of "[FACTS: A Factored State-Space Framework For World Modelling](https://arxiv.org/abs/2410.20922)" (accepted at **ICLR 2025**)\
(Full release coming soon).

![](assets/FACTS.jpg?raw=true)

## Installation:
1.  (Recommended) Install your dependencies, especially [pytorch](https://pytorch.org/):
   ```
        'pytorch>=2.1.0'
        'einops>=0.8.0'
   ``` 
2.  In a terminal, ```cd``` to ```FACTS/```, and run ```pip install -e .```
3. (Optional) Test your installation:\
    Under ```FACTS/```, run
    ```
        . demos/scripts/test_0.sh
        . demos/scripts/test_1.sh
    ```
    If both tests are passed, you are GOOD TO GO!

## Usage:
We provide only three examples to show its usage, for now, more details and [DEMOS](#demos) will be released later. Stay tuned, until then... 
* Example 1 (very basic):
```
    import torch
    from facts_ssm import FACTS

    facts=FACTS(
        in_features=32,
        in_factors=128,  # M
        num_factors=8,  # K
        slot_size=32,  # D
    ).to('cuda')

    X = torch.randn(4, 30, 128, 32).to('cuda')  # [batch, seq_len, M, D]
    y, z = facts(X)   # [batch, seq_len, K, D], [batch, seq_len, K, D]
    print(f"Output y:  {y.size()}")
    print(f"Output z:  {z.size()}")
```  

* Example 2 (flexible customisation):
```
    import torch
    from facts_ssm import FACTS

    facts=FACTS(
        in_features=32,
        in_factors=128,  # M
        num_factors=128,  # K
        slot_size=32,  # D
        num_heads=4,  # multi-head FACTS
        dropout=0.1,  # dropout
        C_rank=32,  # set to D to use the C proj in SSMs
        router='sfmx_attn',  # router customisation  
        init_method='learnable',  # choose a RNN memory init method
        slim_mode=True,  # Turn on to save ~25% params
        residual=True  # Support only M==K
    ).to('cuda')

    X = torch.randn(4, 30, 128, 32).to('cuda')  # [batch, seq_len, M, D]
    y, z = facts(X)   # [batch, seq_len, K, D], [batch, seq_len, K, D]
    print(f"Output y:  {y.size()}")
    print(f"Output z:  {z.size()}")
```


* Example 3 (semi-parallel RNNs, i.e. chunking):
```
    import torch
    from facts_ssm import FACTS

    facts=FACTS(
        in_features=32,
        in_factors=128,  # M
        num_factors=128,  # K
        slot_size=32,  # D
        num_heads=4,  
        dropout=0.1,  
        C_rank=32,  # set to D to allow output proj. C
        fast_mode=False,  # mute full-length parallel scan
        chunk_size=16  # parallel within the chunk, sequential across chunks
    ).to('cuda')

    X = torch.randn(4, 30, 128, 32).to('cuda')  # [batch, seq_len, M, D]
    y, z = facts(X)   # [batch, seq_len, K, D], [batch, seq_len, K, D]
    print(f"Output y:  {y.size()}")
    print(f"Output z:  {z.size()}")
```

## Demos:
Coming soon ...



## Contact
We constantly respond to the raised ''issues'' in terms of running the code. For further inquiries and discussions (e.g. questions about the paper), email: linanbo2008@gmail.com.


## Citation
If you find this code useful, please reference in your paper:
```
@article{nanbo2024facts,
  title={FACTS: A Factored State-Space Framework For World Modelling},
  author={Nanbo, Li and Laakom, Firas and Xu, Yucheng and Wang, Wenyi and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:2410.20922},
  year={2024}
}
```