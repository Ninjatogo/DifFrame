import itertools
import time
from multiprocessing import Pool, cpu_count

import numpy as np

from FrameProcessor import FrameProcessor


class DifframeEngine:
    frame_processor: FrameProcessor

    def __init__(self, in_input_folder):
        self.frame_processor = FrameProcessor(in_input_folder, 0.92)

    def process_video_complete_loop(self):
        print('DifFrame Main')
        frame_range_raw = [y for y in range(len(self.frame_processor.frameInputPaths))]
        frame_range_chunks = np.array_split(frame_range_raw, cpu_count())
        self.frame_processor.set_dicing_rate(2)

        with Pool() as pool:
            pool_output = pool.starmap(self.frame_processor.identify_differences, [(x, True) for x in frame_range_chunks])

        for item in list(itertools.chain(*pool_output)):
            self.frame_processor.frameCollector.dictionary_append(item.FrameIndex, item.FrameX, item.FrameY)

        self.frame_processor.generate_and_save_delta_frames()

        self.frame_processor.reconstruct_frames()


if __name__ == "__main__":
    time0 = time.time()
    loop_count = 1
    difframe_engine = DifframeEngine('SampleFrames-Mob_Psycho_100_BIG')
    for i in range(loop_count):
        difframe_engine.process_video_complete_loop()
    time1 = time.time()
    print(f"Time taken to execute {loop_count} time(s): {time1 - time0} (s)")
