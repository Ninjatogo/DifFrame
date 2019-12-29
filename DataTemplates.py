from collections import namedtuple

diffBlckMin = namedtuple('diffBlckMin', ['FrameData', 'FrameX', 'FrameY'])
diffBlckTransfer = namedtuple('diffBlckTransfer', ['FrameData', 'FrameIndex', 'FrameX', 'FrameY'])
diffBlckComplete = namedtuple('diffBlckComplete', ['FrameX', 'FrameY', 'Filename', 'FileX', 'FileY'])
