import itertools
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from FrameProcessor import FrameProcessor


def main():
    print('DifFrame Main')
    qaz = FrameProcessor('SampleFrames-Mob_Psycho_100', 0.92)
    frame_range_raw = [y for y in range(17)]
    frame_range_chunked = np.array_split(frame_range_raw, cpu_count())
    qaz.setDivisionDimensions(16, 9)

    with Pool() as pool:
        pool_output = pool.starmap(qaz.extractDifferences, [(x, True) for x in frame_range_chunked])

    for item in list(itertools.chain(*pool_output)):
        qaz.frameCollector.dictAppend(item.FrameData, item.FrameIndex, item.FrameX, item.FrameY)

    qaz.generateBatchFrames()


if __name__ == "__main__":
    time0 = time.time()
    for i in range(10):
        main()
    time1 = time.time()
    print(f"Time taken to execute 10 times: {time1 - time0} (s)")
