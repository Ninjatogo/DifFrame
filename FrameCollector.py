import cv2
import numpy as np
import DataTemplates as datTemp


class FrameCollector:
    def __init__(self):
        self.diffBlocksTempDict = {'count': 0}
        self.diffBlocksStorageDict = {'OutputFrame': 0}

    def dictionary_append(self, inFrameNumber, inFrameX, inFrameY):
        if self.diffBlocksTempDict.get(inFrameNumber) is None:
            self.diffBlocksTempDict[inFrameNumber] = []
        self.diffBlocksTempDict[inFrameNumber].append(
            datTemp.diffBlckMin(FrameX=inFrameX, FrameY=inFrameY))
        self.diffBlocksTempDict['count'] += 1

    def is_working_set_ready(self, inWorkingSetSize):
        return self.diffBlocksTempDict['count'] >= inWorkingSetSize

    def get_working_set_collection(self, inWorkingSetSize):
        working_set = {}
        if self.diffBlocksTempDict['count'] >= inWorkingSetSize:
            items_added: int = 0
            keys = list(self.diffBlocksTempDict.keys())

            for key in [key for key in keys if key != 'count']:
                if len(self.diffBlocksTempDict.get(key)) > 0:
                    while self.diffBlocksTempDict.get(key):
                        if items_added < inWorkingSetSize:
                            if working_set.get(key) is None:
                                working_set[key] = []
                            working_set[key].append(self.diffBlocksTempDict.get(key).pop())
                            self.diffBlocksTempDict['count'] -= 1
                            items_added += 1
                        else:
                            break
                if items_added >= inWorkingSetSize:
                    break
        return working_set

    def generate_output_frames(self, inCROP_W_SIZE, inCROP_H_SIZE):
        working_set_dict = self.get_working_set_collection(inCROP_W_SIZE * inCROP_H_SIZE)
        working_set_tuple_list = []
        image_buffer = []
        file_name = self.diffBlocksStorageDict['OutputFrame']

        keys = list(working_set_dict.keys())
        for frameNumber in keys:
            frame_block_list = working_set_dict.get(frameNumber)
            for frameBlock in frame_block_list:
                working_set_tuple_list.append((frameNumber, frameBlock))

        for y in range(inCROP_H_SIZE):
            # Start off image array with one frame block to give loop something to append to
            image_strip = working_set_tuple_list[y * inCROP_W_SIZE][1].FrameData

            if self.diffBlocksStorageDict.get(working_set_tuple_list[y * inCROP_W_SIZE][0]) is None:
                self.diffBlocksStorageDict[working_set_tuple_list[y * inCROP_W_SIZE][0]] = []

            _frameX = working_set_tuple_list[y * inCROP_W_SIZE][1].FrameX
            _frameY = working_set_tuple_list[y * inCROP_W_SIZE][1].FrameY
            frame_diff_block_complete = datTemp.diffBlckComplete(FrameX=_frameX, FrameY=_frameY, Filename=file_name,
                                                                 FileX=0, FileY=y)
            self.diffBlocksStorageDict[working_set_tuple_list[y * inCROP_W_SIZE][0]].append(frame_diff_block_complete)

            for x in range(inCROP_W_SIZE - 1):
                image_strip = np.concatenate(
                    [image_strip, working_set_tuple_list[(x + 1) + (y * inCROP_W_SIZE)][1].FrameData], axis=1)
                if self.diffBlocksStorageDict.get(working_set_tuple_list[(x + 1) + (y * inCROP_W_SIZE)][0]) is None:
                    self.diffBlocksStorageDict[working_set_tuple_list[(x + 1) + (y * inCROP_W_SIZE)][0]] = []

                _frameX = working_set_tuple_list[(x + 1) + (y * inCROP_W_SIZE)][1].FrameX
                _frameY = working_set_tuple_list[(x + 1) + (y * inCROP_W_SIZE)][1].FrameY
                frame_diff_block_complete = datTemp.diffBlckComplete(FrameX=_frameX, FrameY=_frameY, Filename=file_name,
                                                                     FileX=x + 1, FileY=y)
                self.diffBlocksStorageDict[working_set_tuple_list[(x + 1) + (y * inCROP_W_SIZE)][0]].append(
                    frame_diff_block_complete)

            image_buffer.append(image_strip)

        image_buffer2 = image_buffer[0]
        for i in range(len(image_buffer)):
            if (i + 1) < len(image_buffer):
                image_buffer2 = np.concatenate([image_buffer2, image_buffer[i + 1]], axis=0)
        return image_buffer2

    def save_to_disk(self, inCROP_W_SIZE, inCROP_H_SIZE):
        image_buffer2 = self.generate_output_frames(inCROP_W_SIZE, inCROP_H_SIZE)
        cv2.imwrite(f"OutputFrames\\pic{self.diffBlocksStorageDict['OutputFrame']}.jpg", image_buffer2)
        self.diffBlocksStorageDict['OutputFrame'] += 1
