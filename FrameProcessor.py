# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:30:27 2019

@author: Max R. (Ninjatogo)
"""

import os
import concurrent.futures

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from FrameCollector import FrameCollector
import DataTemplates as datTemp
import AspectRatioCalculator


class FrameProcessor:
    frameCollector: FrameCollector

    def __init__(self, in_frame_directory, in_similarity_threshold):
        self.currentFrameIndex = -10
        self.currentFrameData = np.array([[0, 0, 0], [0, 0, 0]], np.uint8)
        self.frameCollector = FrameCollector()
        self.frameDirectoryPath = in_frame_directory
        self.framePaths = []
        self.similarityThreshold = in_similarity_threshold
        self.frameHeight = 1
        self.frameWidth = 1
        self.frameDivisionDimensionX = 1
        self.frameDivisionDimensionY = 1
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

    def generate_output_frames(self, in_crop_w_size, in_crop_h_size):
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

    def save_to_disk(self, in_file_name, in_file_data):
        cv2.imwrite(in_file_name, in_file_data)

    def generate_batch_frames(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while self.frameCollector.is_working_set_ready(self.frameDivisionDimensionX * self.frameDivisionDimensionY):
                image_buffer2 = self.generate_output_frames(self.frameDivisionDimensionX, self.frameDivisionDimensionY)
                file_name = f"OutputFrames\\pic{self.frameCollector.diffBlocksStorageDict['OutputFrame']}.jpg"
                self.frameCollector.diffBlocksStorageDict['OutputFrame'] += 1
                executor.submit(self.save_to_disk, file_name, image_buffer2)

    def identify_differences_single_frame(self, in_file_index, in_return_to_queue):
        difference_list = []
        if in_file_index + 1 < len(self.framePaths):
            frame_a = cv2.imread(self.framePaths[in_file_index])
            frame_b = cv2.imread(self.framePaths[in_file_index + 1])
            gray_frame_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
            gray_frame_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
            for ih in range(self.frameDivisionDimensionY):
                for iw in range(self.frameDivisionDimensionX):
                    x = self.frameWidth / self.frameDivisionDimensionX * iw
                    y = self.frameHeight / self.frameDivisionDimensionY * ih
                    h = self.frameHeight / self.frameDivisionDimensionY
                    w = self.frameWidth / self.frameDivisionDimensionX
                    gray_frame_block_a = gray_frame_a[int(y):int(y + h), int(x):int(x + w)]
                    gray_frame_block_b = gray_frame_b[int(y):int(y + h), int(x):int(x + w)]

                    similarity_score = structural_similarity(gray_frame_block_a, gray_frame_block_b, full=True)[0]

                    if similarity_score <= self.similarityThreshold:
                        if in_return_to_queue:
                            difference_list.append(
                                datTemp.diffBlckTransfer(FrameIndex=in_file_index,
                                                         FrameX=iw, FrameY=ih))
                        else:
                            self.frameCollector.dictionary_append(in_file_index, iw, ih)
        return difference_list

    def identify_differences(self, in_file_indices, in_return_to_queue):
        difference_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_differences = {
                executor.submit(self.identify_differences_single_frame, inFileIndex, in_return_to_queue): inFileIndex
                for inFileIndex in in_file_indices}
            for future in concurrent.futures.as_completed(future_to_differences):
                difference_list.extend(future.result())
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
        with os.scandir(self.frameDirectoryPath) as entries:
            for entry in entries:
                self.framePaths.append(entry.path)
