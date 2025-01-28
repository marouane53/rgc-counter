# src/config.py

MICRONS_PER_PIXEL = 0.5  # <-- Example scale in microns/pixel. 
                         #    Adjust this based on your microscope calibration.

CHANNEL_INDEX = 0  # If your .tif is single-channel, this might be 0. 
                   # If multi-channel, pick the channel with RBPMS signal.
