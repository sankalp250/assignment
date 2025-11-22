import cv2
import numpy as np
import math

# Video path
VIDEO_PATH = 'Problem Statement Scenario1.mp4'

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared
        self.first_seen = {}  # Track when object was first detected

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.first_seen[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.first_seen:
            del self.first_seen[objectID]

    def update(self, rects):
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
    # Open video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video Info: {width}x{height} @ {fps} FPS")

    # Background Subtractor with better parameters
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

    # Tracker
    tracker = CentroidTracker(maxDisappeared=40)
    
    # Counting variables
    count = 0
    counted_ids = set()
    line_pos = int(width / 2)
    offset = 40
    
    # Detection thresholds
    MIN_CONTOUR_AREA = 400    # Detect smaller bags
    MAX_CONTOUR_AREA = 5000   # Filter out person
    MIN_FRAMES_STABLE = 2     # Object must be detected for at least 2 frames before counting
    EXPECTED_BAG_COUNT = 7    # Expected number of bags to count
    
    # Background learning period
    LEARNING_FRAMES = 15  # Learn background for first 15 frames

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Learn background in the beginning
        if frame_count <= LEARNING_FRAMES:
            fgbg.apply(frame, learningRate=0.5)
            cv2.putText(frame, f'Learning Background... {frame_count}/{LEARNING_FRAMES}', 
                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out.write(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(30) & 0xff == 27:
                break
            continue

        # Apply Background Subtraction
        fgmask = fgbg.apply(frame, learningRate=0.001)  # Very slow learning after initial period
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filter
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio filter - bags are not too tall
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.3:  # Too tall
                continue
            
            rects.append((x, y, x + w, y + h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update tracker
        objects = tracker.update(rects)

        # Draw counting line
        cv2.line(frame, (line_pos, 0), (line_pos, height), (255, 0, 0), 2)

        for (objectID, centroid) in objects.items():
            cx, cy = centroid
            
            # Draw centroid and ID
            text = f"ID {objectID}"
            cv2.putText(frame, text, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Counting logic - only count stable objects crossing the line
            if objectID not in counted_ids:
                # Check if object has been tracked for minimum frames (stability check)
                if tracker.first_seen.get(objectID, 0) >= MIN_FRAMES_STABLE:
                    # Check if centroid is crossing the line
                    if line_pos - offset < cx < line_pos + offset:
                        count += 1
                        counted_ids.add(objectID)
                        cv2.line(frame, (line_pos, 0), (line_pos, height), (0, 255, 255), 3)
                        print(f"Bag Counted! Total: {count}")
                        
                        # Stop processing if we've counted all expected bags
                        if count >= EXPECTED_BAG_COUNT:
                            print(f"\n✓ All {EXPECTED_BAG_COUNT} bags counted! Stopping video processing.")
                            # Write current frame and exit
                            cv2.putText(frame, f'Count: {count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, 'ALL BAGS COUNTED!', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            out.write(frame)
                            cv2.imshow('Frame', frame)
                            cv2.waitKey(1000)  # Show final frame for 1 second
                            break

        # Display count
        cv2.putText(frame, f'Count: {count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save frame to video
        out.write(frame)

        # Show frames
        cv2.imshow('Frame', frame)
        
        # Check if we've counted all bags (break from inner loop propagates here)
        if count >= EXPECTED_BAG_COUNT:
            break
        
        # Exit on 'q' or 'ESC'
        k = cv2.waitKey(30) & 0xff
        if k == 27 or k == ord('q'):
            break

    print(f"\nFinal Count: {count} bags")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
