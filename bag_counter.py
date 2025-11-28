import cv2
import numpy as np
import math

# ============================================================
# VIDEO SELECTION - Change this to switch between videos
# ============================================================
VIDEO_PATH = 'Problem Statement Scenario2.mp4'  # Change to 'Problem Statement Scenario1.mp4' for first video
EXPECTED_BAG_COUNT = 16  # Expected number of bags to count (7 for Scenario1, 16 for Scenario2)
OUTPUT_PATH = 'output.mp4'  # Output video file name

class CentroidTracker:
    """Track objects across frames using centroid tracking."""
    
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared
        self.first_seen = {}  # Track when object was first detected

    def register(self, centroid):
        """Register a new object."""
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.first_seen[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """Remove an object from tracking."""
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.first_seen:
            del self.first_seen[objectID]

    def update(self, rects):
        """Update tracked objects with new detections."""
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = []
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(inputCentroids)):
                    dist = math.hypot(objectCentroids[i][0] - inputCentroids[j][0],
                                      objectCentroids[i][1] - inputCentroids[j][1])
                    row.append(dist)
                D.append(row)
            D = np.array(D)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row][col] > 50:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.first_seen[objectID] += 1

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


def main():
    """Main function to process video and count bags."""
    
    # Open video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n{'='*60}")
    print(f"BAG COUNTER")
    print(f"{'='*60}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Resolution: {width}x{height} @ {fps:.2f} FPS")
    
    # Determine video orientation and set counting line
    is_landscape = width > height
    
    if is_landscape:
        line_orientation = 'vertical'
        line_pos = int(width / 2)
        line_start = (line_pos, 0)
        line_end = (line_pos, height)
        offset = 80  # Wider zone for landscape
    else:
        # For portrait - place line in lower third where bags pass
        line_orientation = 'horizontal'
        line_pos = int(height * 0.65)  # Lower third instead of middle
        line_start = (0, line_pos)
        line_end = (width, line_pos)
        offset = 100  # Much wider zone for portrait
    
    print(f"Orientation: {'Landscape' if is_landscape else 'Portrait'}")
    print(f"Counting Line: {line_orientation} at position {line_pos}")
    print(f"{'='*60}\n")

    # Initialize background subtractor with more aggressive settings
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=300,  # Longer history
        varThreshold=12,  # Lower threshold = more sensitive
        detectShadows=False
    )
    tracker = CentroidTracker(maxDisappeared=60)  # Increased from 40
    
    # Counting variables
    count = 0
    counted_ids = set()
    
    # Detection parameters - MUCH more permissive
    if is_landscape:
        MIN_CONTOUR_AREA = 300
        MAX_CONTOUR_AREA = 8000
        MIN_ASPECT_RATIO = 0.3
    else:
        # Portrait video needs different parameters
        MIN_CONTOUR_AREA = 200  # Lower minimum
        MAX_CONTOUR_AREA = 15000  # Higher maximum for larger bags
        MIN_ASPECT_RATIO = 0.2  # More permissive aspect ratio
    
    MIN_FRAMES_STABLE = 1  # Reduced from 2 - count faster
    LEARNING_FRAMES = 20  # More learning frames
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    tracked_boxes = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Learn background in the beginning
        if frame_count <= LEARNING_FRAMES:
            fgbg.apply(frame, learningRate=0.7)  # More aggressive learning
            cv2.putText(frame, f'Learning Background... {frame_count}/{LEARNING_FRAMES}', 
                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out.write(frame)
            cv2.imshow('Bag Counter', frame)
            if cv2.waitKey(30) & 0xff == 27:
                break
            continue

        # Apply background subtraction with lower threshold
        fgmask = fgbg.apply(frame, learningRate=0.0005)  # Very slow learning
        _, fgmask = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)  # Lower threshold

        # More aggressive morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Larger kernel
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        rect_boxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filter
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Less restrictive aspect ratio filter
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < MIN_ASPECT_RATIO:
                continue
            
            rect = (x, y, x + w, y + h)
            rects.append(rect)
            rect_boxes.append((x, y, w, h))

        # Update tracker
        objects = tracker.update(rects)
        
        # Map tracked objects to bounding boxes
        tracked_boxes.clear()
        for objectID, centroid in objects.items():
            cx, cy = centroid
            min_dist = float('inf')
            closest_box = None
            
            for i, (x, y, x2, y2) in enumerate(rects):
                rect_cx = (x + x2) / 2
                rect_cy = (y + y2) / 2
                dist = math.hypot(cx - rect_cx, cy - rect_cy)
                if dist < min_dist:
                    min_dist = dist
                    closest_box = rect_boxes[i]
            
            if closest_box and min_dist < 80:  # Increased from 50
                tracked_boxes[objectID] = closest_box

        # Draw counting line
        cv2.line(frame, line_start, line_end, (255, 0, 0), 3)

        # Count stable objects
        stable_count = sum(1 for oid in objects.keys() if tracker.first_seen.get(oid, 0) >= MIN_FRAMES_STABLE)

        # Process tracked objects
        for objectID, centroid in objects.items():
            cx, cy = centroid
            
            if objectID in tracked_boxes:
                frames_tracked = tracker.first_seen.get(objectID, 0)
                is_stable = frames_tracked >= MIN_FRAMES_STABLE
                is_counted = objectID in counted_ids
                
                # Calculate distance from counting line
                if is_landscape:
                    distance_from_line = abs(cx - line_pos)
                else:
                    distance_from_line = abs(cy - line_pos)
                    
                is_near_line = distance_from_line < 250  # Increased visibility range
                
                # Draw box if stable AND (near line OR counted)
                if is_stable and (is_near_line or is_counted):
                    x, y, w, h = tracked_boxes[objectID]
                    
                    box_color = (0, 255, 255) if is_counted else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    
                    text = f"ID {objectID}"
                    cv2.putText(frame, text, (cx - 10, cy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    cv2.circle(frame, (cx, cy), 4, box_color, -1)

            # Counting logic
            if objectID not in counted_ids:
                if tracker.first_seen.get(objectID, 0) >= MIN_FRAMES_STABLE:
                    is_crossing = False
                    if is_landscape:
                        is_crossing = line_pos - offset < cx < line_pos + offset
                    else:
                        is_crossing = line_pos - offset < cy < line_pos + offset
                    
                    if is_crossing:
                        count += 1
                        counted_ids.add(objectID)
                        cv2.line(frame, line_start, line_end, (0, 255, 255), 5)
                        print(f"✓ Bag Counted! Total: {count}")
                        
                        if count >= EXPECTED_BAG_COUNT:
                            print(f"\n{'='*60}")
                            print(f"✓ All {EXPECTED_BAG_COUNT} bags counted!")
                            print(f"{'='*60}\n")
                            cv2.putText(frame, f'Count: {count}', (30, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, 'ALL BAGS COUNTED!', (30, 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            out.write(frame)
                            cv2.imshow('Bag Counter', frame)
                            cv2.waitKey(1000)
                            break

        # Display count and tracking info
        cv2.putText(frame, f'Count: {count}', (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Tracking: {stable_count}', (30, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Save and show frame
        out.write(frame)
        cv2.imshow('Bag Counter', frame)
        
        if count >= EXPECTED_BAG_COUNT:
            break
        
        if cv2.waitKey(30) & 0xff == 27:
            break

    print(f"Final Count: {count} bags")
    print(f"Output saved to: {OUTPUT_PATH}\n")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

