# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:30:27 2019

@author: Max R. (Ninjatogo)
"""

import cv2
import os
import numpy as np
import multiprocessing
import FrameCollector as framColl
import DataTemplates as datTemp
from skimage.metrics import structural_similarity


class FrameProcessor:
    def __init__(self, inFrameDirectory, inSimilarityThreshold):
        self.currentFrameIndex = 0
        self.frameCollector = framColl.FrameCollector()
        self.frameDirectoryPath = inFrameDirectory
        self.framePaths = []
        self.similarityThreshold = inSimilarityThreshold
        self.frameHeight = 1
        self.frameWidth = 1
        self.frameDivisionDimensionX = 1
        self.frameDivisionDimensionY = 1
        self.loadFilePaths()

    def generateBatchFrames(self):
        while(self.frameCollector.isWorkingSetReady2(self.frameDivisionDimensionX * self.frameDivisionDimensionY)):
            self.frameCollector.saveToDisk2(self.frameDivisionDimensionX, self.frameDivisionDimensionY)

    def extractDifferences(self, inFileIndex):
        if(inFileIndex + 1 < len(self.framePaths)):
            frameA = cv2.imread(self.framePaths[inFileIndex])
            frameB = cv2.imread(self.framePaths[inFileIndex + 1])
            grayFrameA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
            grayFrameB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)
            for ih in range(self.frameDivisionDimensionY):
                for iw in range(self.frameDivisionDimensionX):
                    x = self.frameWidth / self.frameDivisionDimensionX * iw
                    y = self.frameHeight / self.frameDivisionDimensionY * ih
                    h = self.frameHeight / self.frameDivisionDimensionY
                    w = self.frameWidth / self.frameDivisionDimensionX
                    grayFrameBlockA = grayFrameA[int(y):int(y + h), int(x):int(x + w)]
                    grayFrameBlockB = grayFrameB[int(y):int(y + h), int(x):int(x + w)]

                    similarityScore = structural_similarity(grayFrameBlockA, grayFrameBlockB, full=True)[0]

                    if(similarityScore <= self.similarityThreshold):
                        colorFrameBlockB = frameB[int(y):int(y + h), int(x):int(x + w)]
                        colorFrameBlockB = np.pad(colorFrameBlockB, pad_width=((2, 2), (2, 2), (0, 0)), mode='edge')
                        self.frameCollector.dictAppend(colorFrameBlockB, inFileIndex, iw, ih)

    def setDivisionDimensions(self, inDimensionX, inDimensionY):
        if(len(self.framePaths) > 0):
            img = cv2.imread(self.framePaths[0])
            height, width, channels = img.shape
            self.frameWidth = width
            self.frameHeight = height

            # TODO: Loop on dimensions, decrementing the dimension until a value that fits is found
            if(width % inDimensionX == 0):
                self.frameDivisionDimensionX = inDimensionX
            if(height % inDimensionY == 0):
                self.frameDivisionDimensionY = inDimensionY

    def loadFilePaths(self):
        with os.scandir(self.frameDirectoryPath) as entries:
            for entry in entries:
                self.framePaths.append(entry.path)
