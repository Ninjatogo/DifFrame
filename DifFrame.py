import time

from joblib import Parallel, cpu_count, delayed

from FrameProcessor import FrameProcessor


def main():
    print('DifFrame Main')
    qaz = FrameProcessor('SampleFrames-Mob_Psycho_100', 0.92)
    frame_range_raw = [y for y in range(17)]
    qaz.setDivisionDimensions(16, 9)

    with Parallel(n_jobs=cpu_count(), require='sharedmem') as parallel:
        parallel(delayed(qaz.extractDifferences)(i) for i in frame_range_raw)

    qaz.generateBatchFrames()


if __name__ == "__main__":
    time0 = time.time()
    for i in range(10):
        main()
    time1 = time.time()
    print(f"Time taken to execute 10 times: {time1 - time0} (s)")
