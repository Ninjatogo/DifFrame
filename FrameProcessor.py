# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:30:27 2019

@author: Max R. (Ninjatogo)
"""

import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from multiprocessing import Pool, cpu_count

from FrameCollector import FrameCollector
import DataTemplates as datTemp
import AspectRatioCalculator


def save_single_image_to_disk(in_file_name, in_file_data):
    cv2.imwrite(in_file_name, in_file_data)


def save_image_batch_to_disk(in_image_batch):
    for image in in_image_batch:
        save_single_image_to_disk(image[0], image[1])


def scale_frame(in_frame):
    scale_percent = 80

    # calculate the 50 percent of original dimensions
    width = int(in_frame.shape[1] * (scale_percent / 100))
    height = int(in_frame.shape[0] * (scale_percent / 100))

    new_dimensions = (width, height)

    # resize image
    output = cv2.resize(in_frame, new_dimensions)
    return output


class FrameProcessor:
    frameCollector: FrameCollector

    def __init__(self, in_frame_input_directory, in_similarity_threshold):
        self.currentFrameIndex = -10
        self.currentFrameData = np.array([[0, 0, 0], [0, 0, 0]], np.uint8)
        self.frameCollector = FrameCollector()
        self.frameInputDirectoryPath = in_frame_input_directory
        self.framePaths = []
        self.similarityThreshold = in_similarity_threshold
        self.frameHeight = 1
        self.frameWidth = 1
        self.frameDivisionDimensionX = 1
        self.frameDivisionDimensionY = 1
        self.miniBatchSize = 4
        self.load_file_paths()
        self.set_dicing_rate(1)

    def update_loaded_frame(self, in_frame_index):
        if in_frame_index != self.currentFrameIndex:
            self.currentFrameIndex = in_frame_index
            self.currentFrameData = cv2.imread(self.framePaths[in_frame_index + 1])

    def extract_differences(self, in_frame_block_x, in_frame_block_y):
        x = self.frameWidth / self.frameDivisionDimensionX * in_frame_block_x
        y = self.frameHeight / self.frameDivisionDimensionY * in_frame_block_y
        h = self.frameHeight / self.frameDivisionDimensionY
        w = self.frameWidth / self.frameDivisionDimensionX
        color_frame_block_b = self.currentFrameData[int(y):int(y + h), int(x):int(x + w)]
        color_frame_block_b = np.pad(color_frame_block_b, pad_width=((2, 2), (2, 2), (0, 0)),
                                     mode='edge')
        return color_frame_block_b

    def generate_delta_frame(self, in_crop_w_size, in_crop_h_size):
        working_set_dict = self.frameCollector.get_working_set_collection(in_crop_w_size * in_crop_h_size)
        working_set_tuple_list = []
        image_strips = []
        file_name = self.frameCollector.diffBlocksStorageDict['OutputFrame']

        keys = list(working_set_dict.keys())
        for frameNumber in keys:
            frame_block_list = working_set_dict.get(frameNumber)
            for frameBlock in frame_block_list:
                working_set_tuple_list.append((frameNumber, frameBlock))

        for y in range(in_crop_h_size):
            # Start off image array with one frame block to give loop something to append to
            self.update_loaded_frame(working_set_tuple_list[y * in_crop_w_size][1].FrameIndex)
            image_strip = self.extract_differences(working_set_tuple_list[y * in_crop_w_size][1].FrameX,
                                                   working_set_tuple_list[y * in_crop_w_size][1].FrameY)

            if self.frameCollector.diffBlocksStorageDict.get(working_set_tuple_list[y * in_crop_w_size][0]) is None:
                self.frameCollector.diffBlocksStorageDict[working_set_tuple_list[y * in_crop_w_size][0]] = []

            _frameX = working_set_tuple_list[y * in_crop_w_size][1].FrameX
            _frameY = working_set_tuple_list[y * in_crop_w_size][1].FrameY
            frame_diff_block_complete = datTemp.diffBlckComplete(FrameX=_frameX, FrameY=_frameY, Filename=file_name,
                                                                 FileX=0, FileY=y)
            self.frameCollector.diffBlocksStorageDict[working_set_tuple_list[y * in_crop_w_size][0]].append(
                frame_diff_block_complete)

            for x in range(in_crop_w_size - 1):
                self.update_loaded_frame(working_set_tuple_list[(x + 1) + (y * in_crop_w_size)][1].FrameIndex)
                _frame_data = self.extract_differences(
                    working_set_tuple_list[(x + 1) + (y * in_crop_w_size)][1].FrameX,
                    working_set_tuple_list[(x + 1) + (y * in_crop_w_size)][1].FrameY)
                image_strip = np.concatenate([image_strip, _frame_data], axis=1)
                if self.frameCollector.diffBlocksStorageDict.get(
                        working_set_tuple_list[(x + 1) + (y * in_crop_w_size)][0]) is None:
                    self.frameCollector.diffBlocksStorageDict[
                        working_set_tuple_list[(x + 1) + (y * in_crop_w_size)][0]] = []

                _frameX = working_set_tuple_list[(x + 1) + (y * in_crop_w_size)][1].FrameX
                _frameY = working_set_tuple_list[(x + 1) + (y * in_crop_w_size)][1].FrameY
                frame_diff_block_complete = datTemp.diffBlckComplete(FrameX=_frameX, FrameY=_frameY, Filename=file_name,
                                                                     FileX=x + 1, FileY=y)
                self.frameCollector.diffBlocksStorageDict[
                    working_set_tuple_list[(x + 1) + (y * in_crop_w_size)][0]].append(
                    frame_diff_block_complete)

            image_strips.append(image_strip)

        image_buffer = image_strips[0]
        for i in range(len(image_strips)):
            if (i + 1) < len(image_strips):
                image_buffer = np.concatenate([image_buffer, image_strips[i + 1]], axis=0)
        return image_buffer

    def generate_and_save_delta_frames(self):
        temp_batch_collection = []
        while self.frameCollector.is_working_set_ready(self.frameDivisionDimensionX * self.frameDivisionDimensionY):
            image_buffer2 = self.generate_delta_frame(self.frameDivisionDimensionX, self.frameDivisionDimensionY)
            file_name = f"Frames_Deltas\\delta_{self.frameCollector.diffBlocksStorageDict['OutputFrame']}.jpg"
            self.frameCollector.diffBlocksStorageDict['OutputFrame'] += 1
            temp_batch_collection.append((file_name, image_buffer2))
            if len(temp_batch_collection) == cpu_count() * self.miniBatchSize:
                batch_collection_chunks = np.array_split(temp_batch_collection, cpu_count())
                with Pool() as pool:
                    pool.map(save_image_batch_to_disk,
                             [x for x in batch_collection_chunks])
                temp_batch_collection.clear()
        # save any remaining frames on a single CPU
        for item in temp_batch_collection:
            save_single_image_to_disk(item[0], item[1])

    def identify_differences_single_frame(self, in_file_index, in_return_to_queue):
        difference_list = []
        if in_file_index + 1 < len(self.framePaths):
            frame_a = cv2.imread(self.framePaths[in_file_index])
            frame_b = cv2.imread(self.framePaths[in_file_index + 1])
            frame_a_resized = scale_frame(frame_a)
            frame_b_resized = scale_frame(frame_b)
            gray_frame_a = cv2.cvtColor(frame_a_resized, cv2.COLOR_BGR2GRAY)
            gray_frame_b = cv2.cvtColor(frame_b_resized, cv2.COLOR_BGR2GRAY)
            for ih in range(self.frameDivisionDimensionY):
                for iw in range(self.frameDivisionDimensionX):
                    x = gray_frame_a.shape[1] / self.frameDivisionDimensionX * iw
                    y = gray_frame_a.shape[0] / self.frameDivisionDimensionY * ih
                    h = gray_frame_a.shape[0] / self.frameDivisionDimensionY
                    w = gray_frame_a.shape[1] / self.frameDivisionDimensionX
                    gray_frame_block_a = gray_frame_a[int(y):int(y + h), int(x):int(x + w)]
                    gray_frame_block_b = gray_frame_b[int(y):int(y + h), int(x):int(x + w)]
                    #print(f"Image dimensions {gray_frame_block_a.shape}")

                    similarity_score = structural_similarity(gray_frame_block_a, gray_frame_block_b, full=True)[0]

                    if similarity_score <= self.similarityThreshold:
                        if in_return_to_queue:
                            difference_list.append(
                                datTemp.diffBlckTransfer(FrameIndex=in_file_index,
                                                         FrameX=iw, FrameY=ih))
                        else:
                            self.frameCollector.dictionary_append(in_file_index, iw, ih)
        return difference_list

    def identify_differences(self, in_file_indices, in_return_to_queue=True):
        difference_list = []
        for inFileIndex in in_file_indices:
            difference_list.extend(self.identify_differences_single_frame(inFileIndex, in_return_to_queue))
        return difference_list

    def set_dicing_rate(self, in_rate=1):
        if len(self.framePaths) > 0:
            img = cv2.imread(self.framePaths[0])
            height, width, channels = img.shape
            self.frameWidth = width
            self.frameHeight = height

            aspect_ratio = AspectRatioCalculator.calculate_aspect_ratio(self.frameWidth, self.frameHeight)
            self.frameDivisionDimensionX = aspect_ratio[0] * in_rate
            self.frameDivisionDimensionY = aspect_ratio[1] * in_rate

    def load_file_paths(self):
        with os.scandir(self.frameInputDirectoryPath) as entries:
            for entry in entries:
                self.framePaths.append(entry.path)

    def reconstruct_frames(self, in_use_same_directory=True, in_custom_directory=''):
        print('Reconstructing video frames')
        frame_buffer = []
        with os.scandir('Frames_Upscaled') as upscaled_entries:
            entries_hold = []
            for entry in upscaled_entries:
                entries_hold.append(entry.path)
            entries_hold.sort()
            first_scaled_frame = entries_hold[0]
        # Split seed frame into blocks matching original frame processor size with the correct dicing rate
        for frame_key in self.frameCollector.diffBlocksStorageDict:
            print(f'Frame key: {frame_key}')
            # Get data from frame buffer
            # Replace parts of buffer that have been extracted by difference processor
