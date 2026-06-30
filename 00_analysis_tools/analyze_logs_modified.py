# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()


def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {best/acc}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{(((json_log.split("/"))[-2]).split("_"))[-1]}/{(((json_log.split("/"))[-1]).split("_"))[-3]}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[-1]]:  # metric AND epoch, BOTH should be in .json
                raise KeyError(
                    f'last epoch of {args.json_logs[i]} does not contain metric {metric}')
            xs = []
            ys = []

            if metric not in log_dict[epochs[0]]:
                xs.append(np.array([epochs[0]]))
                # ys.append(np.zeros(len(log_dict[epochs[0]]['iter'])))
                ys.append(np.array([np.NAN]))

            for k, epoch in enumerate(epochs):
                iters = log_dict[epoch]['iter']
                if log_dict[epoch]['mode'][-1] == 'val':
                    iters = iters[:-1]  # removing iter of "mode": "val"
                
                if metric in (log_dict[epoch]).keys():
                    # num_iters_per_epoch = iters[-1]  # last index of iter before "mode": "val". For old iter format
                    # xs.append(np.array(iters) + (epoch - 1) * num_iters_per_epoch)  # append current iter value. For old iter format

                    # xs.append(np.array(iters))  # for xs = iter
                    xs.append(np.array([epoch]))  # for xs = epoch

                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))

            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            
            # plt.xlabel('iter')  # for xs = iter
            plt.xlabel('epoch')  # for xs = epoch

            plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        if args.title is not None:
            plt.title(args.title)
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out)
        plt.cla()


def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['top1_acc'],
        help='the metric that you want to plot')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, top1_acc
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field, i.e. not the last iter / not to plot
                if 'epoch' not in log:
                    continue  # "epoch": _, should be in .json
                epoch = log.pop('epoch')
                if epoch not in log_dict:  # if key epoch not yet created in log_dict
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)

    # log_dicts
    # 
    # [
    #  {
    #     1: {
    #          'mode': ['train'], 
    #          'lr': [0.004999785873037446], 
    #          'data_time': [0.0045792579650878905], 
    #          'grad_norm': [1.4281674385070802], 
    #          'loss': [1.4950079053640366], 
    #          'top1_acc': [1.0], 
    #          'top5_acc': [1.0], 
    #          'loss_cls': [1.4950079053640366], 
    #          'time': [0.02739635705947876], 
    #          'iter': [7917], 
    #          'memory': [263], 
    #          'step': [7917]
    #        },
    #     ...
    #     5: {
    #          'mode': ['train', 'val'], 
    #          'lr': [0.004999143420637911], 
    #          'data_time': [0.004415082931518555, 0.005323137641704698], 
    #          'grad_norm': [1.6436187386512757], 
    #          'loss': [1.3695583790540695], 
    #          'top1_acc': [1.0], 
    #          'top5_acc': [1.0], 
    #          'loss_cls': [1.3695583790540695], 
    #          'time': [0.021520829200744628, 0.0134524574496256], 
    #          'iter': [15834, 2], 
    #          'memory': [263], 
    #          'step': [15834, 2],
    #          'acc/top1': [0.37375651350071054], 
    #          'acc/top5': [0.6737723038054635], 
    #          'acc/mean1': [0.13798725350521882]
    #        },
    #     ...
    #  },
    # 
    #  {
    #     1: ...
    #  }
    # ]

    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)


if __name__ == '__main__':
    main()
