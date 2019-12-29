import cv2
import numpy as np
import DataTemplates as datTemp

class FrameCollector:
    def __init__(self):
        self.diffBlocksTempDict = {}
        self.diffBlocksTempDict['count'] = 0
        self.diffBlocksStorageDict = {}
        self.diffBlocksStorageDict['OutputFrame'] = 0

    def dictAppend(self, inFrameData, inFrameNumber, inFrameX, inFrameY):
        if(self.diffBlocksTempDict.get(inFrameNumber) is None):
            self.diffBlocksTempDict[inFrameNumber] = []
        self.diffBlocksTempDict[inFrameNumber].append(datTemp.diffBlckMin(FrameData=inFrameData, FrameX=inFrameX, FrameY=inFrameY))
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
                workingSetTupleList.append((frameNumber, frameBlock))

        for y in range(inCROP_H_SIZE):
            # Start off image array with one frame block to give loop something to append to
            imageStrip = workingSetTupleList[y * inCROP_W_SIZE][1].FrameData

            if(self.diffBlocksStorageDict.get(workingSetTupleList[y * inCROP_W_SIZE][0]) is None):
                self.diffBlocksStorageDict[workingSetTupleList[y * inCROP_W_SIZE][0]] = []

            _frameX = workingSetTupleList[y * inCROP_W_SIZE][1].FrameX
            _frameY = workingSetTupleList[y * inCROP_W_SIZE][1].FrameY
            frameDiffBlockComplete = datTemp.diffBlckComplete(FrameX=_frameX, FrameY=_frameY, Filename=fileName, FileX=0, FileY=y)
            self.diffBlocksStorageDict[workingSetTupleList[y * inCROP_W_SIZE][0]].append(frameDiffBlockComplete)

            for x in range(inCROP_W_SIZE - 1):
                imageStrip = np.concatenate([imageStrip, workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][1].FrameData], axis=1)
                if(self.diffBlocksStorageDict.get(workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][0]) is None):
                    self.diffBlocksStorageDict[workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][0]] = []

                _frameX = workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][1].FrameX
                _frameY = workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][1].FrameY
                frameDiffBlockComplete = datTemp.diffBlckComplete(FrameX=_frameX, FrameY=_frameY, Filename=fileName, FileX=x + 1, FileY=y)
                self.diffBlocksStorageDict[workingSetTupleList[(x + 1) + (y * inCROP_W_SIZE)][0]].append(frameDiffBlockComplete)

            imageBuffer.append(imageStrip)

        imageBuffer2 = imageBuffer[0]
        for i in range(len(imageBuffer)):
            if ((i + 1) < len(imageBuffer)):
                imageBuffer2 = np.concatenate([imageBuffer2, imageBuffer[i + 1]], axis=0)
        return imageBuffer2

    def saveToDisk2(self, inCROP_W_SIZE, inCROP_H_SIZE):
        imageBuffer2 = self.generateOutputFrames2(inCROP_W_SIZE, inCROP_H_SIZE)
        cv2.imwrite(f"OutputFrames\\pic{self.diffBlocksStorageDict['OutputFrame']}.jpg", imageBuffer2)
        self.diffBlocksStorageDict['OutputFrame'] += 1
