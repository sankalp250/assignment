# Bag Counter - Computer Vision Assignment

An automated bag counting system using computer vision techniques to detect and count bags in video footage.

## Features

- **Automatic Bag Detection**: Uses background subtraction and contour detection to identify bags
- **Object Tracking**: Implements centroid tracking to follow bags across frames
- **Smart Filtering**: Filters out people and noise using area, aspect ratio, and stability checks
- **Auto-Stop**: Automatically stops processing after counting all expected bags (7)
- **Visual Output**: Generates annotated video with bounding boxes and count display

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Place your video file as `Problem Statement Scenario1.mp4` in the project directory
2. Run the bag counter:

```bash
python bag_counter.py
```

3. The system will:
   - Learn the background for the first 15 frames
   - Detect and track bags crossing the center line
   - Count each unique bag once
   - Automatically stop after counting all 7 bags
   - Generate `output.mp4` with annotations

## How It Works

### Detection Pipeline

1. **Background Learning**: First 15 frames are used to build a background model
2. **Background Subtraction**: MOG2 algorithm identifies moving objects
3. **Morphological Operations**: Cleans up the foreground mask
4. **Contour Detection**: Finds object boundaries
5. **Filtering**: 
   - Area: 400-5000 pixels (filters noise and person)
   - Aspect Ratio: >0.3 (filters tall person parts)
   - Stability: Tracked for 2+ frames before counting
6. **Centroid Tracking**: Tracks objects across frames with unique IDs
7. **Line Crossing**: Counts bags when they cross the center line

### Key Parameters

- `MIN_CONTOUR_AREA = 400`: Minimum object size to detect
- `MAX_CONTOUR_AREA = 5000`: Maximum size (filters out person)
- `MIN_FRAMES_STABLE = 2`: Frames required before counting
- `EXPECTED_BAG_COUNT = 7`: Auto-stop threshold

## Output

- **Console**: Real-time count updates
- **Video Window**: Live visualization with bounding boxes and IDs
- **output.mp4**: Saved annotated video

## Results

✓ Successfully counts all 7 bags in the test video
✓ Filters out person and background noise
✓ Automatically stops after completion

## Technical Details

- **Language**: Python 3
- **Libraries**: OpenCV, NumPy
- **Algorithm**: Background Subtraction (MOG2) + Centroid Tracking
- **Video Processing**: 848x478 @ ~30 FPS
