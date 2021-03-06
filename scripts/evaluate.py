# -*- coding: utf-8 -*-
"""
Created on 28/12/2021  18:23 

@author: Soan Duong, ORCID: https://orcid.org/0000-0002-2092-0088
"""
import os
import sys
parentdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentdir)

import glob
import numpy as np
import pandas as pd
from utils import get_name_from_path
from utils.tracking import track_evaluate, create_sequence, create_benchmark,\
                    create_tracker_results, summarize, read_txt_to_mot, pair_gt_result

# Set the tracking results for evaluation

RESULT_PATHs = ['./sample/deepsort_frcnn_r18/deepsort_resnext50_32x4d_swsl.txt',
                './sample/deepsort_frcnn_r18/deepsort_mobilenet_v3_large.txt',
                './sample/bytetrack/bytetrack_2_skipframes.txt',
                 './sample/deepsort_frcnn_r18/deepsort_densenet121.txt']
VERBOSE = False
out_file = 'results_rounded.csv'
df_all = []
for RESULT_PATH in RESULT_PATHs:
    print('Tracking result path:', RESULT_PATH)

    # Prepare folders
    SAMPLE_FOLDER = './sample/data/'
    PAIRED_DESTINATION = ''

    os.makedirs(os.path.join(SAMPLE_FOLDER, 'benchmarks'), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_FOLDER, 'seqmaps'), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_FOLDER, 'trackers'), exist_ok=True)

    # Set params of the ground-truth
    GT_PATH = './sample/gt.txt'
    SEQUENCE_NAME = 'cam3_10mins'
    FPS = 25
    SHAPE = (1280,960)
    LENGTH = 14952
    BENCHMARK = 'topview'

    # Extract the ground-truth if results with skip frame scenario
    tmp = RESULT_PATH.split('skipframes')
    if len(tmp) > 1:
        tmp = tmp[0].split('_')
        if tmp[-1] == '':
            SKIP = np.int(tmp[-2])
        else:
            SKIP = np.int(tmp[-1])
        PAIRED_DESTINATION = './sample/paired_gt_result/'

        paired = pair_gt_result(
            gt_path=GT_PATH,
            result_path=RESULT_PATH,
            ref='gt',
            length=LENGTH,
            gt_frame_skip=SKIP,
            destination=PAIRED_DESTINATION,
            posfix='',
            verbose=VERBOSE
        ) # -> paired: {'gt': gt, 'result': result, 'length': new_sequence_length}

        new_sequence_length = paired['length']

        LENGTH = new_sequence_length
        GT_PATH = os.path.join(PAIRED_DESTINATION, os.path.basename(GT_PATH))
        RESULT_PATH = os.path.join(PAIRED_DESTINATION, os.path.basename(RESULT_PATH))

    # Create the benchmark and results directories
    if VERBOSE: print('\n===== Create sequence folder', '='*5)
    cam3_10mins = create_sequence(
        gt_path=GT_PATH,
        name=SEQUENCE_NAME,
        fps=FPS, shape=SHAPE, length=LENGTH,
        verbose=VERBOSE
    )

    if VERBOSE: print('\n===== Create Benchmark folder', '='*5)
    create_benchmark(
        benchmark_name=BENCHMARK,
        sequences=[cam3_10mins],
        destination=os.path.join(SAMPLE_FOLDER, 'benchmarks'),
        seqmap_path=os.path.join(SAMPLE_FOLDER, 'seqmaps'),
        verbose=VERBOSE
    )

    if VERBOSE: print('\n===== Create result folders', '='*5)
    byte_track_result_path = RESULT_PATH
    create_tracker_results(
        result_path=byte_track_result_path,
        benchmark_name=BENCHMARK, sequence_name=SEQUENCE_NAME, tracker_name=get_name_from_path(RESULT_PATH),
        destination=os.path.join(SAMPLE_FOLDER, 'trackers'),
        verbose=VERBOSE
    )

    # Evaluate
    gt_topview = os.path.join(SAMPLE_FOLDER, 'benchmarks', BENCHMARK)
    tracker_topview = os.path.join(SAMPLE_FOLDER, 'trackers', BENCHMARK)
    seqmap_file = os.path.join(SAMPLE_FOLDER, 'seqmaps', f'{BENCHMARK}.txt')

    results = track_evaluate(benchmark=BENCHMARK,
                             gt_folder=gt_topview,
                             trackers_folder=tracker_topview,
                             seqmap_file=seqmap_file)
    tracker_name, df = summarize(results, verbose=VERBOSE)[-1] # get last 

    df = df.drop(labels=SEQUENCE_NAME, axis=0)
    df.insert(0, 'detection_results', [tracker_name], allow_duplicates=False)
    df_all.append(df)
    
    import shutil
    shutil.rmtree(SAMPLE_FOLDER, ignore_errors=True)
    if PAIRED_DESTINATION: shutil.rmtree(PAIRED_DESTINATION, ignore_errors=True)

df_all = pd.concat(df_all)

for c in df_all.columns[1:4]:
    df_all[c] = df_all[c].apply(lambda x: round(x, 4))
df_all.to_csv(out_file, index=False)
print(f'Results: {os.path.abspath(out_file)}')