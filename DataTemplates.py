from collections import namedtuple

diffBlckMin = namedtuple('diffBlckMin', ['FrameX', 'FrameY'])
diffBlckTransfer = namedtuple('diffBlckTransfer', ['FrameIndex', 'FrameX', 'FrameY'])
diffBlckComplete = namedtuple('diffBlckComplete', ['FrameX', 'FrameY', 'Filename', 'FileX', 'FileY'])
