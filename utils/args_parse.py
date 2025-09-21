# This module cannot import any other PyTorch/XLA module. Only Python core modules.
import argparse
import os
import sys


def parse_common_options(logdir=None,
                         num_cores=None,
                         global_batch_size=256,
                         epochs=1,
                         num_workers=8,
                         log_steps=1,
                         lr=1e-4,
                         momentum=None,
                         profiler_port=9012,
                         opts=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument('--logdir', type=str, default=logdir)
    parser.add_argument('--num_cores', type=int, default=num_cores)
    parser.add_argument('--global_batch_size', type=int, default=global_batch_size)
    parser.add_argument('--num_epochs', type=int, default=epochs)
    parser.add_argument(
       '--num_steps',
       type=int,
       help='used for testing for the test to run shorter.')
    parser.add_argument('--num_workers', type=int, default=num_workers)
    parser.add_argument('--log_steps', type=int, default=log_steps)
    parser.add_argument('--profiler_port', type=int, default=profiler_port)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--tidy', action='store_true')
    parser.add_argument('--metrics_debug', action='store_true')
    parser.add_argument('--async_closures', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)

    if opts:
        for name, aopts in opts:
            parser.add_argument(name, **aopts)
    args, leftovers = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + leftovers
    # Setup import folders.
    xla_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    sys.path.append(os.path.join(os.path.dirname(xla_folder), 'test'))
    return args