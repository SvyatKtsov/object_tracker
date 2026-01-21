# Optimized CPU-based Object Tracking with Python multiprocessing library 
this is an implementation of CPU-bound optimization for three opencv object trackers: MIL, KCF and CSRF

## Usage instructions
### 1. Install all dependencies:
	pip install -r requirements.txt
### 2. Run .py file:
	python track_objects.py [tracker_number] [use_mask] [video_stream]
where the 1st argument "tracker_number" is the number of tracker, integer(1-3);
2nd argument is bool value in the form of integer(0,1): 0 - don't use mask for showing only the tracked area and 1 - use mask for showing full frames;
3rd argument may be either int (0..., number of video stream (web camera)) or string - path to video in any format (/path/video)

## Usage Example
	python 3 0 0
	python 1 1 folder1/video.mp4



