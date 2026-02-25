##### attempt to filter double peaked peaks #####

# wave show function in lib rosa (plots hist of audio data)
# let this run for whole dataset and get info of in which files and times double peaks are -> make histogram

# import matplotlib.pyplot as plt
import re
from pathlib import Path
from audioio import load_audio
from scipy.signal import find_peaks
import numpy as np
from IPython import embed
from matplotlib import pyplot as plt


wavpath = Path("/home/eisele/labdata/eels-mfn2021/recordings2024-02-14/")

# Match eellogger followed by any number, then -YYYYMMDDTHHMMSS
pattern = re.compile(r"^eellogger\d+-\d{8}T\d{6}$")

# Get all wav files of directory and store them alphabetically in a list
wav_path_list = sorted(
    [
        f
        for f in wavpath.glob("*.wav")
        # only store if normal stem and not empty
        if pattern.match(f.stem) and f.stat().st_size > 0
    ]
)

# load all wav files and store in one variable
# wav_data = []
# for path in track(
#     wav_path_list, total=len(wav_path_list), description="Loading wav files..."
# ):
#     data = list(AudioLoader(path))
#     wav_data.append(data)


# find way to extract double peaks with find peaks function
# put this into loop once it works for whole recording day
signal, rate = load_audio(str(wav_path_list[1]))
time = np.arange(signal.shape[0]) / rate


### find peaks
peaks, properties = find_peaks(
    signal[:, 0],
    height=0.0003,
    width=(57.6, 120.0),
)

peak_values = []
segment_len = 2.5 * (rate / 1000)  # ms * rate per ms

### get data around detected peaks
for i, peak in enumerate(peaks):
    # skip peaks that are too close together
    if peak - peaks[i - 1] < segment_len / 2:
        continue
    peak_segment = np.arange(int(peak - segment_len / 2), int(peak + segment_len / 2))
    # check if segment is out of bounds
    if peak_segment[0] < 0 or peak_segment[-1] >= signal.shape[0]:
        continue  # skip if segment is out of bounds
    # peak_values_mean_ch = np.mean(signal[peak_segment, :], axis=1)

    peak_values_mean_ch = signal[peak_segment, 0]
    # filter some out to make plot übersichtlicher
    if (
        peak_values_mean_ch[int(segment_len / 2)] < 0.0001
        and peak_values_mean_ch[int(segment_len / 2)] > -0.0001
    ):
        continue
    # average over all channels
    peak_values.append(peak_values_mean_ch)


### plot
for i in peak_values:
    plt.plot(i)
plt.axvline(x=segment_len / 2, color="black")
plt.show()

embed()
