# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 18:30:27 2019

@author: Max R. (Ninjatogo)
"""

import time
import itertools
import cv2
import os
import numpy as np
import multiprocessing
from skimage.metrics import structural_similarity
from collections import namedtuple


differenceBlockMinimum = namedtuple('differenceBlockMinimum', ['FrameData','FrameX','FrameY'])
differenceBlockProcessTransfer = namedtuple('differenceBlockProcessTransfer', ['FrameData','FrameIndex','FrameX','FrameY'])
differenceBlockComplete = namedtuple('differenceBlockComplete', ['FrameX','FrameY','Filename','FileX','FileY'])

class FrameCollector:
    def __init__(self):
        self.diffBlocksTempDict = {}
        self.diffBlocksTempDict['count'] = 0
        self.diffBlocksStorageDict = {}
        self.diffBlocksStorageDict['OutputFrame'] = 0
            
    def dictAppend(self, inFrameData, inFrameNumber, inFrameX, inFrameY):
        with lock:
            if(self.diffBlocksTempDict.get(inFrameNumber) is None):
                self.diffBlocksTempDict[inFrameNumber] = []
        self.diffBlocksTempDict[inFrameNumber].append(differenceBlockMinimum(FrameData = inFrameData, FrameX = inFrameX, FrameY = inFrameY))
        self.diffBlocksTempDict['count'] += 1
    
    def isWorkingSetReady2(self, inWorkingSetSize):
        return (self.diffBlocksTempDict['count'] >= inWorkingSetSize)
        
    def getWorkingSetCollection2(self, inWorkingSetSize):
        workingSet = {}
        if(self.diffBlocksTempDict['count'] >= inWorkingSetSize):
            itemsAdded = 0
            keys = list(self.diffBlocksTempDict.keys())
            
            for key in [key for key in keys if key != 'count']:
                if(len(self.diffBlocksTempDict.get(key)) > 0):
                    while self.diffBlocksTempDict.get(key):
                        if(itemsAdded < inWorkingSetSize):                              
                            if(workingSet.get(key) is None):
                                workingSet[key] = []
                            with lock:
                                workingSet[key].append(self.diffBlocksTempDict.get(key).pop())
                                self.diffBlocksTempDict['count'] -= 1
                            itemsAdded += 1
                        else:
                            break
                if(itemsAdded >= inWorkingSetSize):
                    break
        return workingSet

    def generateOutputFrames2(self, inCROP_W_SIZE, inCROP_H_SIZE):
        workingSetDict = self.getWorkingSetCollection2(inCROP_W_SIZE * inCROP_H_SIZE)
        workingSetTupleList = []
        imageBuffer = []
        fileName = self.diffBlocksStorageDict['OutputFrame']

        keys = list(workingSetDict.keys())
        for frameNumber in keys:
            frameBlockList = workingSetDict.get(frameNumber)
            for frameBlock in frameBlockList:
                workingSetTupleList.append((frameNumber,frameBlock))
        
        for y in range(inCROP_H_SIZE):
            # Start off image array with one frame block to give loop something to append to
            imageStrip = workingSetTupleList[y * inCROP_W_SIZE][1].FrameData

            with lock:
                if(self.diffBlocksStorageDict.get(workingSetTupleList[y * inCROP_W_SIZE][0]) is None):
                    self.diffBlocksStorageDict[workingSetTupleList[y * inCROP_W_SIZE][0]] = []

            _frameX = workingSetTupleList[y * inCROP_W_SIZE][1].FrameX
            _frameY = workingSetTupleList[y * inCROP_W_SIZE][1].FrameY
            frameDiffBlockComplete = differenceBlockComplete(FrameX = _frameX, FrameY = _frameY, Filename = fileName, FileX = 0, FileY = y)
            self.diffBlocksStorageDict[workingSetTupleList[y * inCROP_W_SIZE][0]].append(frameDiffBlockComplete)

            for x in range(inCROP_W_SIZE - 1):
                imageStrip = np.concatenate([imageStrip, workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][1].FrameData], axis=1)
                with lock:
                    if(self.diffBlocksStorageDict.get(workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][0]) is None):
                        self.diffBlocksStorageDict[workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][0]] = []

                _frameX = workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][1].FrameX
                _frameY = workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][1].FrameY
                frameDiffBlockComplete = differenceBlockComplete(FrameX = _frameX, FrameY = _frameY, Filename = fileName, FileX = x + 1, FileY = y)
                self.diffBlocksStorageDict[workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][0]].append(frameDiffBlockComplete)

            imageBuffer.append(imageStrip)
    
        imageBuffer2 = imageBuffer[0]
        for i in range(len(imageBuffer)):
            if ((i + 1) < len(imageBuffer)):
                imageBuffer2 = np.concatenate([imageBuffer2, imageBuffer[i + 1]], axis=0)        
        return imageBuffer2

    def saveToDisk2(self, inCROP_W_SIZE, inCROP_H_SIZE):
        imageBuffer2 = self.generateOutputFrames2(inCROP_W_SIZE, inCROP_H_SIZE)
        cv2.imwrite(f"Cropped\\pic{self.diffBlocksStorageDict['OutputFrame']}.jpg",imageBuffer2)
        with lock:
            self.diffBlocksStorageDict['OutputFrame'] += 1
        
            
class FrameProcessor:
    def __init__(self, inFrameDirectory, inSimilarityThreshold):
        self.currentFrameIndex = 0
        self.frameCollector = FrameCollector()
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
    
    def extractDifferences(self, inFileIndices, inReturnToQueue):
        #print(f'Beginning image diff-split process for frame {inFileIndex}')
        differenceList = []
        for inFileIndex in inFileIndices:
            if(inFileIndex + 1 < len(self.framePaths)):
                frameA = cv2.imread(self.framePaths[inFileIndex])
                frameB = cv2.imread(self.framePaths[inFileIndex+1])
                grayFrameA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
                grayFrameB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)
                for ih in range(self.frameDivisionDimensionY):
                    for iw in range(self.frameDivisionDimensionX):
                        x = self.frameWidth / self.frameDivisionDimensionX * iw 
                        y = self.frameHeight / self.frameDivisionDimensionY * ih
                        h = self.frameHeight / self.frameDivisionDimensionY
                        w = self.frameWidth / self.frameDivisionDimensionX
                        grayFrameBlockA = grayFrameA[int(y):int(y+h), int(x):int(x+w)]
                        grayFrameBlockB = grayFrameB[int(y):int(y+h), int(x):int(x+w)]
        
                        similarityScore = structural_similarity(grayFrameBlockA, grayFrameBlockB, full=True)[0]
        
                        if(similarityScore <= self.similarityThreshold):
                            colorFrameBlockB = frameB[int(y):int(y+h), int(x):int(x+w)]
                            colorFrameBlockB = np.pad(colorFrameBlockB, pad_width=((2, 2), (2, 2), (0, 0)), mode='edge')
                            if(inReturnToQueue == True):
                                differenceList.append(differenceBlockProcessTransfer(FrameData=colorFrameBlockB, FrameIndex=inFileIndex, FrameX=iw, FrameY=ih))
                            else:
                                self.frameCollector.dictAppend(colorFrameBlockB, inFileIndex, iw, ih)
        return differenceList
                
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
                

def init_child(lock_):
    global lock
    lock = lock_

def main():
    print('DifFrame Main')
    qaz = FrameProcessor('Mob Psycho Image Sequence', 0.92)
    frameRangeRaw = [y for y in range(17)]
    cpuCount = multiprocessing.cpu_count()
    frameRangeChunked = np.array_split(frameRangeRaw, cpuCount)
    qaz.setDivisionDimensions(16,9)
    lock = multiprocessing.Lock()
    init_child(lock)
    # Single Thread
    #for x in range(17):
    #    qaz.extractDifferences([x], False)
    # Single Thread
    # Multiple Threads
    poolOutput = []
    with multiprocessing.Pool(initializer=init_child, initargs=(lock,)) as pool:
        poolOutput = pool.starmap(qaz.extractDifferences, [(x, True) for x in frameRangeChunked])

    for item in list(itertools.chain(*poolOutput)):
        qaz.frameCollector.dictAppend(item.FrameData, item.FrameIndex, item.FrameX, item.FrameY)
    # Multiple Threads
    qaz.generateBatchFrames()
        
if __name__ == "__main__":
    time0 = time.time()
    for i in range(10):
        main()
    time1 = time.time()
    print(time1 - time0)