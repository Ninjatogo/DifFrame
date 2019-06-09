# DifFrame
Pack video frame differences into a single frame for efficient video upscaling.
 DifFrame is meant to be used as a video pre-processor to make video upscaling using machine learning-based tools faster. The way I intend to do this is by comparing each video frame with the next frame to find the differences. Once the difference is found, it can be extracted and copied to a collection. Once the collection has enough parts to make a full frame it can be saved to disk to be upscaled.
After the difference collection is upscaled, the frame can be divided again to produce the individual frame pieces that will need to be copied back to the video.
