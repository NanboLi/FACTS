import os, sys
import torch
from demos.utils import FlexibleArgumentParser
from facts_ssm import FACTS


def get_parser():
    parser = FlexibleArgumentParser(description='FACTS')

    # basic config
    parser.add_argument('--debugging', type=int, required=False, default=0, help='status')

    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='FACTS',
                        help='model name, e.g. FACTS, iTransformer')

    # I/O specs
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--log_path', type=str, default='./logs', help='where to save the logs')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # permutation invariance test
    parser.add_argument('--permute_input', type=int, default=0, help='permutation invariance test')

    # FACTS-specific
    parser.add_argument('--graph_embed', type=str, default='ms_conv2d',
                        help='method of series graph embedding, options:[dft, conv2d, ms_conv2d]')
    parser.add_argument('--router', type=str, default='sfmx_attn',
                        help='dynamic routers for FACTS')
    parser.add_argument('--init_method', type=str, default='init_state',
                        help='memory initialisation method for FACTS')
    parser.add_argument('--decoder', type=str, default='fgd',
                        help='decoding mechanism for FACTS, one of [fgd, sbd]')
    parser.add_argument('--num_factors', type=int, default=7, help='number of factors for the FACTS')
    parser.add_argument('--slot_size', type=int, default=8, help='dimension of factors for the FACTS')
    parser.add_argument('--chunk_size', type=int, default=1, 
                        help='chunk the sequence for recursive FACTS computation')
    parser.add_argument('--slim_mode', type=int, default=0, 
                        help='whether to activate slim mode for the FACTS module')
    parser.add_argument('--fast_approx', type=int, default=1, 
                        help='whether to activate fast mode for the FACTS module')
    parser.add_argument('--facts_residual', type=int, default=1, 
                        help='whether to use residual connection for the FACTS module')
    parser.add_argument('--autoregressive', type=int, default=0, 
                        help='whether to use autoregressive predictions')

    # general model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba and FACTS decoder')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba and FACTS')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--rev_in', type=int, default=1, 
                        help='whether to rev_in for the sequence standarization')
 
    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
  
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
        fast_mode=(args.fast_approx==1),
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
    print(f" >>>>>>>>>> Unit test: {os.path.basename(__file__)} passed! <<<<<<<<<< ")
            

if __name__ == '__main__':
    main()
    sys.exit()