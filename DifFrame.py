import FrameProcessor as framProc
from joblib import Parallel, cpu_count, delayed
import time
import multiprocessing
import itertools
import numpy as np

def main():
    print('DifFrame Main')
    qaz = framProc.FrameProcessor('SampleFrames-Mob_Psycho_100', 0.92)
    frameRangeRaw = [y for y in range(17)]
    qaz.setDivisionDimensions(16, 9)

    with Parallel(n_jobs=cpu_count(), require='sharedmem') as parallel:
        parallel(delayed(qaz.extractDifferences)(i) for i in frameRangeRaw)

    qaz.generateBatchFrames()


if __name__ == "__main__":
    time0 = time.time()
    for i in range(10):
        main()
    time1 = time.time()
    print(f'Time taken to execute 10 times: {time1 - time0} (s)')
