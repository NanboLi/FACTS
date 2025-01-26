import os, sys
# print(sys.path)
from demos.utils import FlexibleArgumentParser, prt_config


def get_parser():
    parser = FlexibleArgumentParser(description='FACTS')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

    args_in = parser.parse_args()

    return args_in


def main():
    args = get_parser()
    prt_config(args)
    print(f" >>>>>>>>>> Unit test: {os.path.basename(__file__)} passed! <<<<<<<<<< ")
            

if __name__ == '__main__':
    main()
    sys.exit()