"""
FACTS for Multivariate Time Series (MTS) Tasks: FACTS_MTS.

The structure of FACTS_MTS is adapted from the TSLib code base:
    https://github.com/thuml/Time-Series-Library
Though due to version changes, the code below might not be directly 
compatible with the original TSLib anymore. We will try to close the 
gap in our future release.

Any use of this code should cite the following paper:
    FACTS: A Factored State-Space Framework For World Modelling
    https://arxiv.org/abs/2410.20922

Author: Nanbo Li (linanbo2008@gmail.com)
"""

import torch
import torch.nn as nn
from facts_ssm import FACTS
from facts_ssm.demos.time_series import RevIN, SetEmbedder, Encoder, EncoderLayer, FactorGraphDecoder 


class FACTS_MTS(nn.Module):
    def __init__(self, configs, eps=1e-5):
        super(FACTS_MTS, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len  # temporal dims
        self.M = configs.enc_in  # spatial dims
        self.K = configs.num_factors
        self.slot_size = configs.slot_size
        self.eps = eps

        # preprocessing layers - DataEmbedding
        self.rev_in = RevIN(configs.enc_in, self.eps) if configs.rev_in else None
        self.embedder = SetEmbedder(configs)

        # FACTS encoder
        if configs.mk_setup == 'mmk':
            n_facts_arr = [self.M] * (configs.e_layers) + [self.K]
            n_init_method = ['self'] * (configs.e_layers-1) + [configs.init_method]
        else:
            n_facts_arr = [self.M, self.K] + (configs.e_layers-1) * [self.K]
            n_init_method = [configs.init_method] + (configs.e_layers-1) * ['self']
        self.encoder = Encoder(
            [
                EncoderLayer(
                    facts_module=FACTS(
                        in_features=configs.slot_size,
                        in_factors=n_facts_arr[el],
                        num_factors=n_facts_arr[el+1],
                        slot_size=configs.slot_size,
                        expand=configs.expand,
                        d_conv=configs.d_conv,
                        num_heads=configs.n_heads,
                        dropout=configs.dropout,
                        C_rank=configs.slot_size,
                        router=configs.router,
                        init_method=n_init_method[el],
                        fast_mode=(configs.fast_mode==1) if el==(configs.e_layers-1) else True,
                        slim_mode=(configs.slim_mode==1),
                        residual=(n_facts_arr[el]==n_facts_arr[el+1]),
                        chunk_size=configs.chunk_size
                    ), 
                    expand=configs.expand,
                    num_heads=configs.n_heads,
                    dropout=configs.dropout,
                    norm_first=True,
                    mode="normal"                    
                ) for el in range(configs.e_layers)
            ],
        )
        self.norm_rep = nn.LayerNorm(configs.slot_size)

        # prediction proj
        self.pred_proj = nn.Conv2d(
            configs.seq_len, 
            configs.pred_len,
            bias=True,
            kernel_size=(1, 1)
        )

        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            if configs.decoder == 'fgd':
                self.decoder = FactorGraphDecoder(
                    slot_size=configs.slot_size,
                    out_factors=configs.enc_in,
                    out_dim=configs.pred_len,
                    d_model=configs.d_model,
                    elem_wise_alpha=(configs.elem_wise_decoding==1),
                )
                print('Using FactorGraphDecoder')
            else:
                raise ValueError('Decoder type not supported')

            self.res_pred = nn.Conv1d(
                configs.seq_len, 
                configs.pred_len,
                kernel_size=1, 
                bias=False
            )

    def forecast(self, x_enc, x_mark_enc, permute_input=False):
        if self.rev_in is not None:
            x_enc = self.rev_in(x_enc, 'norm')
        else:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + self.eps)
            x_enc /= stdev
        
        x_res = x_enc

        # Summarize spatial-temporal features as ST graphs
        x_enc = self.embedder(x_enc) # [B, T, M, D]

        # Input permutation exps
        if permute_input:
            perm = torch.randperm(x_enc.size(-2))  # Permute the input variates
            x_enc = x_enc[:, :, perm]

        # FACTS encoding and prediction
        x_enc, _ = self.encoder(x_enc) # [B, T, K, D], wrapper encoder
        x_enc = self.pred_proj(x_enc)  # [B, T, K, D] -> [B, L, K, D]
        x_enc = self.norm_rep(x_enc)

        # Decoding/mixing + residual (noise) 
        x_out = self.decoder(x_enc) + self.res_pred(x_res)  # [B L M]

        if self.rev_in is not None:
            x_out = self.rev_in(x_out, 'denorm')
        else:
            # De-Normalization from Non-stationary Transformer
            x_out = x_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            x_out = x_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, permute_input=False):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc, permute_input=permute_input)
            return x_out[:, -self.pred_len:, :]
        return None