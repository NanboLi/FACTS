"""
This test checks if the installation of FACTS was successful.

Any use of this code should cite the following paper:
    FACTS: A Factored State-Space Framework For World Modelling

Author: Nanbo Li (linanbo2008@gmail.com)
"""
import os, sys
import argparse
import torch
from facts_ssm import FACTS
from facts_ssm.utils import *


def prt_config(cfg):
    for key, value in vars(cfg).items():
        if not key.startswith('__') and not key.endswith('__'):
            print(f' - {key}: {value}')


def override_config_with_args(cfg, args):
    for key, value in vars(args).items():
        setattr(cfg, key, value)
    return cfg


def override_args_with_config(args, cfg):
    """
    Override or adds the attributes of an argparse.Namespace with the attributes of python object.

    Args:
        args (argparse.Namespace): argparse.Namespace object.
        cfg (object): python object.   
    
    Returns:
        argparse.Namespace: argparse.Namespace object.
    """
    for attr_name in dir(cfg):
        if not attr_name.startswith('__') and not callable(getattr(cfg, attr_name)):
            setattr(args, attr_name, getattr(cfg, attr_name))
    return args


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ 
    A custom ArgumentParser that allows for unknown arguments to be passed in and added to the namespace.

    Example:
    ```
    parser = FlexibleArgumentParser(description='Flexible Argument Parser')
    parser.add_argument('--task_name', type=str, default='time_series', help='task name')
    parser.add_argument('--model', type=int, default='FACTS', help='status')

    args = parser.parse_args()
    # prt_config(args)
    ```
    """
    def parse_args(self, args=None, namespace=None):
        # Parse known arguments
        namespace, unknown_args = self.parse_known_args(args, namespace)
        
        # Dynamically add unknown arguments to the namespace
        if unknown_args:
            it = iter(unknown_args)
            for arg in it:
                if arg.startswith("--"):
                    key = arg.lstrip("--")
                    value = next(it, None)
                    setattr(namespace, key, value)
        return namespace


def get_parser():
    parser = FlexibleArgumentParser(description='FACTS')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--seq_len', type=int, default=100, help='sequence length')
    parser.add_argument('--enc_in', type=int, default=64, help='num. of input variates') 

    parser.add_argument('--num_factors', type=int, default=64, help='number of factors for FACTS')
    parser.add_argument('--slot_size', type=int, default=16, help='dimension of factors for FACTS')
    parser.add_argument('--slim_mode', type=int, default=1, help='activate slim mode for FACTS?')
    parser.add_argument('--fast_mode', type=int, default=1, help='activate fast mode for FACTS')
    parser.add_argument('--chunk_size', type=int, default=-1, help='chunk-wise parallel FACTS, -1 for fully parallel')
    parser.add_argument('--init_method', type=str, default='learnable', help='memory init method for FACTS')
    parser.add_argument('--router', type=str, default='sfmx_attn', help='dynamic routers for FACTS')

    parser.add_argument('--expand', type=int, default=2, help='dimension expansion')
    parser.add_argument('--d_conv', type=int, default=2, help='d_conv as in Mamba')
    parser.add_argument('--n_heads', type=int, default=4, help='num. of heads for multi-head models')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    args_in = parser.parse_args()

    return args_in


def main():
    args = get_parser()

    facts_module=FACTS(
        in_features=args.slot_size,
        in_factors=args.enc_in,
        num_factors=args.num_factors,
        slot_size=args.slot_size,
        expand=args.expand,
        d_conv=args.d_conv,
        num_heads=args.n_heads,
        dropout=args.dropout,
        C_rank=args.slot_size,
        router=args.router,
        init_method=args.init_method,
        fast_mode=(args.fast_mode==1),
        slim_mode=(args.slim_mode==1),
        residual=(args.enc_in==args.num_factors),
        chunk_size=args.chunk_size
    )
    print(facts_module)
    print(f"\n")

    if torch.cuda.is_available():
        facts_module.to('cuda')
        X = torch.randn(args.batch_size, args.seq_len, args.enc_in, args.slot_size).to('cuda')
    else:
        facts_module.to('cpu')
        X = torch.randn(args.batch_size, args.seq_len, args.enc_in, args.slot_size).to('cpu')
    
    y, z = facts_module(X)
    print(f"Input Size:  {X.size()}")
    print(f"Output Size:  y: {y.size()},     z: {z.size()}\n")
    print(f" >>>>>>>>>> Unit test: {os.path.basename(__file__)} passed! Installation Successful! <<<<<<<<<< ")
            

if __name__ == '__main__':
    main()
    sys.exit()