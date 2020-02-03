import DataTemplates as datTemp


class FrameCollector:
    def __init__(self):
        self.diffBlocksTempDict = {'count': 0}
        self.diffBlocksStorageDict = {'OutputFrame': 0}

    def dictionary_append(self, in_frame_number, in_frame_block_x, in_frame_block_y):
        if self.diffBlocksTempDict.get(in_frame_number) is None:
            self.diffBlocksTempDict[in_frame_number] = []
        self.diffBlocksTempDict[in_frame_number].append(
            datTemp.diffBlckTransfer(FrameIndex=in_frame_number, FrameX=in_frame_block_x, FrameY=in_frame_block_y))
        self.diffBlocksTempDict['count'] += 1

    def is_working_set_ready(self, in_working_set_size):
        return self.diffBlocksTempDict['count'] >= in_working_set_size

    def get_working_set_collection(self, in_working_set_size):
        working_set = {}
        if self.diffBlocksTempDict['count'] >= in_working_set_size:
            items_added: int = 0
            keys = list(self.diffBlocksTempDict.keys())

            for key in [key for key in keys if key != 'count']:
                if len(self.diffBlocksTempDict.get(key)) > 0:
                    while self.diffBlocksTempDict.get(key):
                        if items_added < in_working_set_size:
                            if working_set.get(key) is None:
                                working_set[key] = []
                            working_set[key].append(self.diffBlocksTempDict.get(key).pop())
                            self.diffBlocksTempDict['count'] -= 1
                            items_added += 1
                        else:
                            break
                if items_added >= in_working_set_size:
                    break
        return working_set
