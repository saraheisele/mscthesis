"This script synchronizes the video and audio data from the electric eels in berlin."

import matplotlib.pyplot as plt
import numpy as np

# load data (first only a single file of each (audio and video))
# load script from patrick that detects LED in video
# change it so it only looks in the relevant pixels 
# see if it can detect LED blinking
# check within first 5 min (verify this number w patrick)
# get frames/timestamps of video file where LED is on
# align with timestamps from audio file that log when LED is being turned off/on
