# Dif-Frame
*Pack video frame differences into a single frame for efficient video upscaling.*

**Dif-Frame** is meant to be used as a video pre-processor to make video upscaling using machine learning-based tools faster. The way this is intend to work is by comparing each video frame with the subsequent frame to find the differences. Once the difference is found, it can be extracted and copied to a collection. In order to reduce the total amount of data that will be copied, the frames are divided into zones. Only zones that have a difference above a set threshold will be copied to the upscaling collection. Once the collection has enough parts to make a full frame it can be saved to disk to be upscaled.

After the difference collection is upscaled, the frame can be divided again to produce the individual frame pieces that can be pasted back into the video by superimposing the pieces over time.