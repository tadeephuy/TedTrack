# Description:
Wrapper of [TrackEval](https://github.com/JonathonLuiten/TrackEval) with added utils in [`utils.py`](https://github.com/tadeephuy/TrackEval/blob/main/utils.py) for ease of use.

# Usage:
Please refer to [`demo.ipynb`](https://github.com/tadeephuy/TrackEval/blob/main/demo.ipynb)

## 0.Prepare folder directories


```python
import os

SAMPLE_FOLDER = './sample/data/'
PAIRED_DESTINATION = ''

os.makedirs(os.path.join(SAMPLE_FOLDER, 'benchmarks'))
os.makedirs(os.path.join(SAMPLE_FOLDER, 'seqmaps'))
os.makedirs(os.path.join(SAMPLE_FOLDER, 'trackers'))
```

## 1. Import functions and specify configs


```python
from utils import track_evaluate, create_sequence, create_benchmark,\
                    create_tracker_results, summarize, read_txt_to_mot

GT_PATH = './sample/gt.txt'
SEQUENCE_NAME = 'cam3_10mins'
FPS = 25
SHAPE = (1280,960)
LENGTH = 14952

BENCHMARK = 'topview'

RESULT_PATH = './sample/bytetrack/bytetrack.txt'
```

## 2. Run this cell in skip frame scenario

We specify `SKIP=3` and `ref='gt'`. This means the groundthuth is sampled after `3` consecutive frames.


```python
from utils import pair_gt_result

SKIP = 3
RESULT_PATH = f'./sample/bytetrack/bytetrack_{SKIP}skipframes.txt'
PAIRED_DESTINATION = './sample/paired_gt_result/'

paired = pair_gt_result(
    gt_path=GT_PATH, 
    result_path=RESULT_PATH,
    ref='gt',
    length=LENGTH,
    gt_frame_skip=SKIP,
    destination=PAIRED_DESTINATION,
    posfix='',
    verbose=True
) # -> paired: {'gt': gt, 'result': result, 'length': new_sequence_length}

new_sequence_length = paired['length']

LENGTH = new_sequence_length
GT_PATH = os.path.join(PAIRED_DESTINATION, os.path.basename(GT_PATH))
RESULT_PATH = os.path.join(PAIRED_DESTINATION, os.path.basename(RESULT_PATH))
```

    Use 'gt' to align and reindex frame id
    Truncate both file at frame id: 14952. Skip 3 frames and reindex to new length of: 3738
    Paired groundtruth is created at: ./sample/paired_gt_result/gt.txt
    Paired result is created at: ./sample/paired_gt_result/bytetrack_3skipframes.txt


## 3. Create the benchmark and results directories.


```python
print('\n===== Create sequence folder', '='*5)
cam3_10mins = create_sequence(
    gt_path=GT_PATH,
    name=SEQUENCE_NAME,
    fps=FPS, shape=SHAPE, length=LENGTH,
    verbose=True
)

print('\n===== Create Benchmark folder', '='*5)
create_benchmark(
    benchmark_name=BENCHMARK, 
    sequences=[cam3_10mins], 
    destination=os.path.join(SAMPLE_FOLDER, 'benchmarks'),
    seqmap_path=os.path.join(SAMPLE_FOLDER, 'seqmaps'),
    verbose=True
)

print('\n===== Create result folders', '='*5)

byte_track_result_path = RESULT_PATH 
create_tracker_results(
    result_path=byte_track_result_path,
    benchmark_name=BENCHMARK, sequence_name=SEQUENCE_NAME, tracker_name='ByteTrack', 
    destination=os.path.join(SAMPLE_FOLDER, 'trackers'), 
    verbose=True
)
```

    
    ===== Create sequence folder =====
    {'name': 'cam3_10mins', 'imDir': '', 'frameRate': 25, 'seqLength': 3738, 'imWidth': 1280, 'imHeight': 960, 'imExt': ''}
    
    ===== Create Benchmark folder =====
    Benchmark 'topview' created at: ./sample/data/benchmarks/topview
    Sequences: ['cam3_10mins']
    Sequences mappings created at: ./sample/data/seqmaps/topview.txt
    
    ===== Create result folders =====
    Result file of 'ByteTrack' for sequence 'cam3_10mins' of benchmark 'topview' is created at: ./sample/data/trackers/topview/ByteTrack/cam3_10mins.txt


## 4. Evaluation


```python
gt_topview = os.path.join(SAMPLE_FOLDER, 'benchmarks', BENCHMARK)
tracker_topview = os.path.join(SAMPLE_FOLDER, 'trackers', BENCHMARK)
seqmap_file = os.path.join(SAMPLE_FOLDER, 'seqmaps', f'{BENCHMARK}.txt')


results = track_evaluate(benchmark=BENCHMARK, 
                         gt_folder=gt_topview, 
                         trackers_folder=tracker_topview, 
                         seqmap_file=seqmap_file)

summarize(results)
```

    
    Evaluating 1 tracker(s) on 1 sequence(s) for 1 class(es) on MotChallenge2DBox dataset using the following metrics: HOTA, CLEAR, Identity, Count
    
    
    Evaluating ByteTrack
    
    ==================== SUMMARY ====================
    ByteTrack



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HOTA</th>
      <th>MOTA</th>
      <th>IDF1</th>
      <th>IDsw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cam3_10mins</th>
      <td>0.418976</td>
      <td>0.473314</td>
      <td>0.573725</td>
      <td>27</td>
    </tr>
    <tr>
      <th>COMBINED_SEQ</th>
      <td>0.418976</td>
      <td>0.473314</td>
      <td>0.573725</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
</div>

## 5. Clean up if needed.


```python
import shutil

shutil.rmtree(SAMPLE_FOLDER, ignore_errors=True)
shutil.rmtree(PAIRED_DESTINATION, ignore_errors=True)
```
