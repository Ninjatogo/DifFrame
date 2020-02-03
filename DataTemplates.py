from collections import namedtuple

diffBlckTransfer = namedtuple('diffBlckTransfer', ['FrameIndex', 'FrameX', 'FrameY'])
diffBlckComplete = namedtuple('diffBlckComplete', ['FrameX', 'FrameY', 'Filename', 'FileX', 'FileY'])
