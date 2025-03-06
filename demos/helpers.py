import os
import argparse


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
    

