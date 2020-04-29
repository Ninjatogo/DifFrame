import itertools
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from FrameProcessor import FrameProcessor


def main():
    print('DifFrame Main')
    frame_processor = FrameProcessor('SampleFrames-Mob_Psycho_100', 0.92)
    frame_range_raw = [y for y in range(len(frame_processor.framePaths))]
    frame_range_chunks = np.array_split(frame_range_raw, cpu_count())
    frame_processor.set_dicing_rate(2)

    with Pool() as pool:
        pool_output = pool.starmap(frame_processor.identify_differences, [(x, True) for x in frame_range_chunks])

    for item in list(itertools.chain(*pool_output)):
        frame_processor.frameCollector.dictionary_append(item.FrameIndex, item.FrameX, item.FrameY)

    frame_processor.generate_and_save_delta_frames()

    frame_processor.reconstruct_frames()


if __name__ == "__main__":
    time0 = time.time()
    loop_count = 1
    for i in range(loop_count):
        main()
    time1 = time.time()
    print(f"Time taken to execute {loop_count} time(s): {time1 - time0} (s)")
