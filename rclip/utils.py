import argparse
import os
import pathlib
import sys

from matplotlib import ticker

import config

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def scatterImage(paths, similarities, zoom=0.03, jenks_breaks=3, size=(50,50)):
    fig, ax = plt.subplots(figsize=size)

    x, y = list(range(0, len(similarities))), similarities

    ax.scatter(x, y)

    for x0, y0, path in zip(reversed(x), reversed(y), reversed(paths)):
        ab = AnnotationBbox(
            OffsetImage(plt.imread(path), zoom=zoom),
            (
                x0,
                y0
                #y0 + (y[0]-y[-1]) / (len(similarities)/(zoom*75)) # shift images above dots
             ),
            frameon=False
        )
        ax.add_artist(ab)

    if jenks_breaks > 0:
        breaks = get_jenks_breaks(data_list=similarities, number_classes=jenks_breaks)

    ax.set_ylabel('Similarity', rotation=90)
    ax.set_xlabel('Images')

    ax2 = ax.twinx()
    ax2.set_ylabel('Cluster', rotation=90)
    ax2.set_ylim(ax.get_ylim())
    ax2.yaxis.set_ticks(breaks)
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    #ax2.tick_params(axis='y', rotation=90)

    for i, line in enumerate(breaks):
        #ax.plot([line for _ in range(len(similarities))], 'k')
        plt.axhline(line, color='k', linestyle='-')
        #pos = len(similarities) if i > len(similarities)//2 else len(similarities)//10
        #plt.text(pos, line, f'cluster {i}', fontsize=30, va='center', ha='center', backgroundcolor='w')

    plt.tight_layout()
    return fig

# https://stackoverflow.com/questions/28416408/scikit-learn-how-to-run-kmeans-on-a-one-dimensional-array
def get_jenks_breaks(data_list, number_classes):
    data_list = data_list.copy()

    data_list.sort()
    mat1 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_classes + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(len(data_list) + 1):
        temp = []
        for j in range(number_classes + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, number_classes + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, number_classes + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(number_classes + 1):
        kclass.append(min(data_list))
    kclass[number_classes] = float(data_list[len(data_list) - 1])
    count_num = number_classes
    while count_num >= 2:  # print "rank = " + str(mat1[k][count_num])
        idx = int((mat1[k][count_num]) - 2)
        # print "val = " + str(data_list[idx])
        kclass[count_num - 1] = data_list[idx]
        k = int((mat1[k][count_num] - 1))
        count_num -= 1
    return kclass

def get_system_datadir() -> pathlib.Path:
    '''
    Returns a parent directory path
    where persistent application data can be stored.

    # linux: ~/.local/share
    # macOS: ~/Library/Application Support
    # windows: C:/Users/<USER>/AppData/Roaming
    '''

    home = pathlib.Path.home()

    if sys.platform == 'win32':
        return home / 'AppData/Roaming'
    elif sys.platform.startswith('linux'):
        return home / '.local/share'
    elif sys.platform == 'darwin':
        return home / 'Library/Application Support'

    raise NotImplementedError(f'"{sys.platform}" is not supported')


def get_app_datadir() -> pathlib.Path:
    app_datadir = os.getenv('DATADIR')
    if app_datadir:
        app_datadir = pathlib.Path(app_datadir)
    else:
        app_datadir = get_system_datadir() / config.NAME
    os.makedirs(app_datadir, exist_ok=True)
    return app_datadir


def top_arg_type(arg: str) -> int:
    arg_int = int(arg)
    if arg_int < 1:
        raise argparse.ArgumentTypeError('number of results to display should be >0')
    return arg_int


def init_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('query')
    parser.add_argument('--dir', '-d', type=str, default="../../frames", help='directory to search')
    parser.add_argument('--top', '-t', type=top_arg_type, default=10, help='number of top results to display')
    parser.add_argument('--filepath-only', '-f', action='store_true', default=False, help='outputs only filepaths')
    parser.add_argument(
        '--skip-index', '-n',
        action='store_true',
        default=False,
        help='don\'t attempt image indexing, saves time on consecutive runs on huge directories'
    )
    parser.add_argument(
        '--exclude-dir',
        action='append',
        help='dir to exclude from search, can be specified multiple times;'
             ' adding this argument overrides the default of ("@eaDir", "node_modules", ".git");'
             ' WARNING: the default will be removed in v2'
    )
    return parser
