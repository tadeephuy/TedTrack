import os
import sys
import pickle as pkl
import pandas as pd
import numpy as np
import torch
from IPython.display import display
from fastprogress import progress_bar
from IPython.display import display
from TrackEval import trackeval
from . import ifnone, get_name_from_path, assert_in_list

__all__ = [
    'track_evaluate', 'create_sequence', 'create_benchmark',
    'create_tracker_results', 'summarize', 'read_txt_to_mot', 'convert_pickled_to_txt',
    'pair_gt_result'
]

def track_evaluate(benchmark, gt_folder, trackers_folder, seqmap_file):
    """
    Main function to evaluate
    
    a.Arguments:
        - benchmark: benchmark name
        - gt_folder: path to benchmark groundtruth folder
        - trackers_folder: path to trackers result folder
        - seqmap_file: path to file which specifies the sequences to be evaluate.
    
    b.Usage:
        Follow these steps:
        0. [optional] use `pair_gt_result` function to pair gt file and result file by frame id and reindex them.
        1. use `create_sequence` function to create sequence label object from groundtruth txt file.
        2. a benchmark can contain many sequences. So create all of the necessary sequences and put into a `created_sequences` list.
        3. [optional] create a directory at `path_to_the_seqmap_folder` for `create_benchmark` function to create the seqmap file later (if necessary or you can create your own seqmap file).
        4. use `create_benchmark` function to create a benchmark that contains the created sequences. pass sequences=`created_sequences`, seqmap_path=`path_to_the_seqmap_folder` (or None if you created your own seqmap file).
        5. use `create_tracker_results` to create result folder from result file. Create as many folder as the number of trackers needed to evaluate.
        6. use this function. Return `results`. Detailed results are also created at each tracker's folder.
        7. use `summarize` function to summarize `results`.
    """
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    config['BENCHMARK'] = benchmark
    config['GT_FOLDER'] = gt_folder 
    config['TRACKERS_FOLDER'] = trackers_folder 
    config['SEQMAP_FILE'] = seqmap_file
    config['PRINT_CONFIG'] = False
    config['PRINT_RESULTS'] = False
    config['TIME_PROGRESS'] = False

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in list(default_metrics_config.keys()) + ['PRINT_CONFIG']}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    results = evaluator.evaluate(dataset_list, metrics_list)
    return results

def read_txt_to_mot(path, length=None, skip=None, frame_id_col=0):
    """
    Load txt file and add '-1' columns to match MOT format
    
    a.Arguments:
        - path: path to txt file
        - length: length of the sequence to evaluate. Leave as is if not specified.
        - skip: number of frames to skip, always start at 1.
    
    b. Flow:
        - load txt path.
        - add '-1' columns for 10 columns in total.
        - truncate the result at the frame length is specified.
        - skip frame if specified.
    """
    df = pd.read_csv(path, header=None)
    if len(df.columns) < 10:
        df[[c for c in range(len(df.columns), 10)]] = -1
    if length is not None:
        df = df.loc[df[0] <= length, :]
    if skip is not None:
        df = skip_frame_id(df, skip=skip, frame_id_col=frame_id_col)
    return df

def convert_pickled_to_txt(pkl_path, destination=None, return_df=False, 
                           frame_id_starts_at_0=True, 
                           length=None, skip=None,
                           verbose=False):
    """
    Read pickled detections to mot txt file
    
    1.Arguments:
        - pkl_path: path to pkl file
        - destination: path to write the csv file
        - return_df: return dataframe on RAM, overrides `destination`
        - frame_id_starts_at_0: if True, increase frame_id by 1 to match MOT formats that starts at 1.
        - length: truncate at frame_id length (MOT format starts at 1)
        - skip: skip frame scenario
    """
    with open(pkl_path, 'rb') as f:
        f = pkl.load(f)
    
    detections = []
    for frame_id, boxes in f:
        boxes = pd.DataFrame(boxes[0], columns=['x1', 'y1', 'x2', 'y2', 'conf'])
        boxes['frame_id'] = frame_id
        detections.append(boxes)
    detections = pd.concat(detections)
    detections['track_id'] = -1
    detections['w'] = detections['x2'] - detections['x1']
    detections['h'] = detections['y2'] - detections['y1']
    
    detections = detections[['frame_id', 'track_id', 'x1', 'y1', 'w', 'h', 'conf']].sort_values(by='frame_id').reset_index(drop=True)
    if frame_id_starts_at_0:
        detections['frame_id'] += 1
        if verbose: print('Increased frame ID by 1 to match MOT format.')
    
    if len(detections.columns) < 10:
        detections[[c for c in range(len(detections.columns), 10)]] = -1
    if length is not None:
        detections = detections.loc[detections[0] <= length, :]
    if skip is not None:
        detections = skip_frame_id(detections, skip=skip, frame_id_col='frame_id')

    if return_df:
        detections.columns = list(range(10))
        return detections

    destination = ifnone(destination, os.path.dirname(pkl_path))
    txt_path = os.path.join(destination, f'{get_name_from_path(pkl_path)}.txt')
    detections.to_csv(txt_path, header=None, index=None)
    if verbose:
        print(f'Converted txt file saved at: {os.path.abspath(txt_path)}')

def create_benchmark(benchmark_name, sequences, destination, seqmap_path=None, verbose=False):
    """
    Create new benchmark folder

    a. Format:
        BENCHMARK
        |___sequence1
        |   |___gt.txt
        |   |___seqinfo.ini
        |
        |___sequence2
            |___gt.txt
            |___seqinfo.ini
    
    b. Arguments:
        - benchmark_name: name of benchmark to create
        - sequences: list of sequence label objects
        - destination: path to create the benchmark
    
    c. Flow:
        - create folder with `benchmark_name` at `destination`.
        - loop over the sequence label objects, for each sequence:
            - create a sequence folder.
            - create groundtruth folder and groundtruth file: gt/gt.txt.
            - create seqinfo.ini file, containings information of the video:
                [Sequence]
                name=cam3_10mins
                imDir=
                frameRate=25
                seqLength=14952
                imWidth=1280
                imHeight=960
                imExt=
    """

    benchmark_path = os.path.join(destination, benchmark_name)
    os.makedirs(benchmark_path, exist_ok=True)

    sequence_name_list = []
    for i, sequence in enumerate(sequences):
        sequence_name = sequence['seqinfo'].get('name', f'sequence_{i}')
        sequence_path = os.path.join(benchmark_path, sequence_name)
        os.makedirs(sequence_path, exist_ok=True)
        sequence['gt'].to_csv(os.path.join(sequence_path, 'gt.txt'), header=None, index=None)
        with open(os.path.join(sequence_path, 'seqinfo.ini'), 'w') as f:
            f.write('[Sequence]')
            for k, v in sequence['seqinfo'].items():
                line = f'{k}={v}\n'
                f.write(line)

        sequence_name_list.append(sequence_name)

    if seqmap_path is not None: 
        seqmap_path = os.path.join(seqmap_path, f'{benchmark_name}.txt')
        with open(seqmap_path, 'w') as f: 
            f.write('name\n')
            for sequence_name in sequence_name_list:
                f.write(f'{sequence_name}\n')

    if verbose:
        print(f'Benchmark \'{benchmark_name}\' created at: {os.path.abspath(benchmark_path)}')
        print(f'Sequences: {[c for c in sequence_name_list]}')
        if seqmap_path is not None:
            print(f'Sequences mappings created at: {os.path.abspath(seqmap_path)}')


def create_sequence(gt_path, fps, shape, name=None, length=None, verbose=False):
    """
    Create a sequence object from a groundtruth file and sequence info

    a. Arguments:
        - gt_path: groundtruth txt path
        - fps: frames per second.
        - shape: (width, height).
        - name: sequence name. Default None -> use name of gt file if not specified.
        - length: total frames in gt. Default None -> use max frame id of gt file if not specified.

    b. Flow:
        - load gt text file.
        - verify and edit text file to gt format if possible.
        - create a seqinfo dictionary: {
            'name':name, 'imDir':'', 'frameRate':fps,
            'seqLength':length, 'imWidth':width,
            'imHeight':height, 'imExt':''
        }
        - return {'gt': gt, 'seqinfo': seqinfo}
    """
    gt = read_txt_to_mot(gt_path)

    seqinfo = {
        'name': ifnone(name, get_name_from_path(gt_path)),
        'imDir': '',
        'frameRate': int(fps),
        'seqLength': ifnone(length, int(gt[0].max())),
        'imWidth': shape[0],
        'imHeight': shape[1],
        'imExt': ''
    }

    if verbose:
        print(seqinfo)
    return {'gt': gt, 'seqinfo': seqinfo}

def create_tracker_results(result_path, benchmark_name, sequence_name, tracker_name=None, destination='./trackers', verbose=False):
    """
    create a tracker result folder

    a. Format: destination/benchmark_name/tracker_name/sequence_name
    b. Arguments:
        - result_path: path to txt result file
        - benchmark_name: benchmark to evaluate
        - sequence_name: name of the input sequence
        - tracker_name: name of the tracker. Use txt result file name if not specified
        - destination: path to create result folder
    
    c. Flow:
        - load result text file.
        - verify and edit text file if possible.
        - create tracker_name folder
        - write the result to the folder
    """
    result = read_txt_to_mot(result_path)
    tracker_name = ifnone(tracker_name, get_name_from_path(result_path))

    result_path = os.path.join(destination, benchmark_name, tracker_name)
    os.makedirs(result_path, exist_ok=True)
    result_path = os.path.join(result_path, f'{sequence_name}.txt')
    result.to_csv(result_path, header=None, index=None)

    if verbose:
        print(f'Result file of \'{tracker_name}\' for sequence \'{sequence_name}\' of benchmark \'{benchmark_name}\' is created at: {os.path.abspath(result_path)}')

def summarize(results, group_by_sequence=False, verbose=True):
    """
    Summarize return results from `track_evaluate`.
    """
    if not verbose:
        def print(*args): return None
        def display(*args): return None

    df_results = []
    print('='*20, 'SUMMARY', '='*20)
    for tracker, result in results[0]['MotChallenge2DBox'].items():
        metric_dict = {}
        if not group_by_sequence: print(tracker)
        for sequence, sequence_result in result.items():
            hota_value = sequence_result['pedestrian']['HOTA']['HOTA'].mean()

            clear_results = sequence_result['pedestrian']['CLEAR']
            mota_value = clear_results['MOTA']
            idsw_value = clear_results['IDSW']

            idf1_value = sequence_result['pedestrian']['Identity']['IDF1']

            metric_dict[sequence] = {
                'HOTA': hota_value, 'MOTA': mota_value, 'IDF1': idf1_value, 'IDsw': idsw_value, 
            }
        df_result = pd.DataFrame(metric_dict).T.astype({'IDsw': int})

        if not group_by_sequence: display(df_result)
        df_results.append((tracker, df_result))
        if not group_by_sequence: print('='*50)

    if group_by_sequence:
        all_sequences = {k: [] for k in list(df_results[0][1].index)}
        for result in df_results:
            name ,df = result
            for i,r in df.iterrows():
                d = pd.DataFrame({name: r.to_dict()}).T.astype({'IDsw': int})
                all_sequences[i].append(d[list(r.keys())])
        for k,v in all_sequences.items():
            print(k)
            v = pd.concat(v)
            display(v)

    return df_results

def reindex_frame_id(file, frame_id_col=0):
    result_file = file.sort_values(by=frame_id_col).reset_index(drop=True)
    frame_id_map = {c:i+1 for i,c in enumerate(np.unique(result_file[frame_id_col].values))}
    result_file[frame_id_col] = result_file[frame_id_col].apply(frame_id_map.get)
    return result_file

def align_to_ref(file, frame_ref, reindex=False, frame_id_col=0):
    result_file = file.loc[file[frame_id_col].isin(frame_ref), :]
    if reindex:
        return reindex_frame_id(file=result_file, frame_id_col=frame_id_col)
    return result_file.sort_values(by=frame_id_col).reset_index(drop=True)

def skip_frame_id(file, skip=1, frame_id_col=0):
    skip_func = lambda x: x%(skip+1)==1
    file = file[file[frame_id_col].apply(skip_func)].reset_index(drop=True)
    return file

def pair_gt_result(gt_path, result_path, ref='gt', 
                   length=None, gt_frame_skip=None, 
                   destination=None, posfix='', frame_id_col=0, 
                   verbose=False):
    """
    Pair gt file and result file by frame id and reindex them, useful for evaluating in different fps/frame skipping scenario.
    
    a.Arguments:
        - gt_path: groundtruth txt path
        - result_path: result txt path
        - ref: ['result', 'gt'] choose either result or groundtruth as the reference frame id
        - length: truncate the result and groundtuth at the frame length if specified
        - gt_frame_skip: skip frame for gt, only when ref='gt'
        - destination: write 2 new file as destination if specified
        - posfix: posfix for the two written file
    
    b. Flow:
        - load gt and result file, verify and truncate, skip frame if specified.
        - create the frame id reference.
        - align the gt and result file to the reference.
        - reindex frame id of the gt and result file as incremental (1,2,3,..) instead of original frame id reference.
        - write to destination is specified.
        - return {'gt': gt, 'result': result}
    """
    assert_in_list(ref, ['result', 'gt'], name='reference')    

    
    result = read_txt_to_mot(result_path, length=length, frame_id_col=frame_id_col)

    if ref=='result':
        gt = read_txt_to_mot(gt_path, length=length, skip=None, frame_id_col=frame_id_col) # only skip frame for gt if gt is reference
        frame_ref = result.loc[:, frame_id_col].values
    else:
        gt = read_txt_to_mot(gt_path, length=length, skip=gt_frame_skip, frame_id_col=frame_id_col) 
        frame_ref = gt.loc[:, frame_id_col].values

    gt = align_to_ref(file=gt, frame_ref=frame_ref, reindex=True, frame_id_col=frame_id_col)
    result = align_to_ref(file=result, frame_ref=frame_ref, reindex=True, frame_id_col=frame_id_col)

    if destination is not None:
        os.makedirs(destination, exist_ok=True)
        paired_gt_path = os.path.join(destination, f'{get_name_from_path(gt_path)}{posfix}.txt')
        paired_result_path = os.path.join(destination, f'{get_name_from_path(result_path)}{posfix}.txt')
        gt.to_csv(paired_gt_path, header=None, index=None)
        result.to_csv(paired_result_path, header=None, index=None)
    
    new_sequence_length = gt[frame_id_col].max()
    if verbose:
        print(f'Use \'{ref}\' to align and reindex frame id')
        if length is not None:
            print(f'Truncate both file at frame id: {length}. Skip {ifnone(gt_frame_skip, 0)} frames and reindex to new length of: {new_sequence_length}')
        if destination is not None:
            print(f'Paired groundtruth is created at: {os.path.abspath(paired_gt_path)}')
            print(f'Paired result is created at: {os.path.abspath(paired_result_path)}')
            

    return {'gt': gt, 'result': result, 'length': new_sequence_length}

