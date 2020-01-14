# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:30:27 2019

@author: Max R. (Ninjatogo)
"""

import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from FrameCollector import FrameCollector
import DataTemplates
import AspectRatioCalculator


class FrameProcessor:
    frameCollector: FrameCollector

    def __init__(self, in_frame_directory, in_similarity_threshold):
        self.currentFrameIndex = 0
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

    def generate_batch_frames(self):
        while self.frameCollector.is_working_set_ready(self.frameDivisionDimensionX * self.frameDivisionDimensionY):
            self.frameCollector.save_to_disk(self.frameDivisionDimensionX, self.frameDivisionDimensionY)

    def extract_differences(self, in_file_indices, in_return_to_queue):
        difference_list = []
        for inFileIndex in in_file_indices:
            if inFileIndex + 1 < len(self.framePaths):
                frame_a = cv2.imread(self.framePaths[inFileIndex])
                frame_b = cv2.imread(self.framePaths[inFileIndex + 1])
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
                            color_frame_block_b = frame_b[int(y):int(y + h), int(x):int(x + w)]
                            color_frame_block_b = np.pad(color_frame_block_b, pad_width=((2, 2), (2, 2), (0, 0)),
                                                         mode='edge')
                            if in_return_to_queue:
                                difference_list.append(
                                    DataTemplates.diffBlckTransfer(FrameData=color_frame_block_b,
                                                                   FrameIndex=inFileIndex,
                                                                   FrameX=iw, FrameY=ih))
                            else:
                                self.frameCollector.dictionary_append(color_frame_block_b, inFileIndex, iw, ih)
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
