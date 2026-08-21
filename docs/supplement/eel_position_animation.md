# Eel head-position animation (electronic supplement)

## Direct links (repo is public; default branch is `master`)

- GIF (QR target):
  https://github.com/saraheisele/mscthesis/blob/master/docs/latex_thesis/figures/position_estimation/eel_position_animation.gif
- MP4:
  https://github.com/saraheisele/mscthesis/blob/master/docs/latex_thesis/figures/position_estimation/eel_position_animation.mp4

## Channel selection (print vs GIF)

- **GIF / MP4:** the right-hand raw panel follows each pulse's head electrode
  (channel with the strongest positive peak at that time; 0-based indices
  matching the electrode ticks).
- **Print keyframes:** both raw panels use fixed electrode 0 so amplitude and
  polarity can be compared on one shared channel.

## Print figure

The thesis PDF shows keyframes from a short turnaround scene in
`eellogger02-20260128T163148.wav` (Berlin tank, dark-area logger).

- Keyframe strip:
  https://raw.githubusercontent.com/saraheisele/mscthesis/master/docs/latex_thesis/figures/position_estimation/eel_position_keyframes.png

## Regenerating the QR code

```bash
EEL_USE_DUMMY_DATASET=0 python code/position_estimation/animate_eel_position.py --make-print-figure
```

`ANIMATION_SUPPLEMENT_URL` in `code/position_estimation/animate_eel_position.py`
must use the `master` branch name (not `main`).
