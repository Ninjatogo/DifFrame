import itertools
import time
from tinydb import TinyDB, Query
from multiprocessing import Pool, cpu_count

import numpy as np

from FrameProcessor import FrameProcessor


def main():
    print('DifFrame Main')
    qaz = FrameProcessor('SampleFrames-Mob_Psycho_100', 0.92)
    frame_range_raw = [y for y in range(17)]
    frame_range_chunked = np.array_split(frame_range_raw, cpu_count())
    qaz.set_dicing_rate(2)

    with Pool() as pool:
        pool_output = pool.starmap(qaz.extract_differences, [(x, True) for x in frame_range_chunked])

    for item in list(itertools.chain(*pool_output)):
        qaz.frameCollector.dictionary_append(item.FrameData, item.FrameIndex, item.FrameX, item.FrameY)

    qaz.generate_batch_frames()


if __name__ == "__main__":
    time0 = time.time()
    loop_count = 10
    for i in range(loop_count):
        main()
    time1 = time.time()
    print(f"Time taken to execute {loop_count} time(s): {time1 - time0} (s)")
