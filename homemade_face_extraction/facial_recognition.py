import cv2
import os
import numpy as np
import mediapipe as mp

def detect_faces_in_video(video_path, output_dir):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.15)  # Lowered confidence for smaller faces

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    face_id = 0  # Counter for face images
    detected_faces_embeddings = []  # Store embeddings to avoid duplicates
    frame_skip = 5  # Balance between speed and accuracy

    def get_face_embedding(face_image):
        # Convert the face to grayscale and resize
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (32, 32), interpolation=cv2.INTER_AREA)  # Smaller size for faster processing
        return resized_face.flatten() / 255.0  # Normalize pixel values

    def is_new_face(embedding, existing_embeddings, threshold=0.4):
        for existing_embedding in existing_embeddings:
            similarity = np.dot(embedding, existing_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(existing_embedding)
            )
            if similarity > threshold:
                return False
        return True

    def detect_faces_at_scales(frame, scales):
        all_faces = []
        for scale in scales:
            scaled_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            results = face_detection.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = scaled_frame.shape
                    x = max(0, int(bboxC.xmin * iw / scale))
                    y = max(0, int(bboxC.ymin * ih / scale))
                    w = int(bboxC.width * iw / scale)
                    h = int(bboxC.height * ih / scale)

                    # Ensure bounding box is within frame boundaries
                    x2 = min(x + w, frame.shape[1])
                    y2 = min(y + h, frame.shape[0])

                    all_faces.append((x, y, x2, y2))
        return all_faces

    frame_count = 0
    while True:
        # Read frame by frame
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames

        # Detect faces at multiple scales
        scales = [1.0, 1.5, 2.0]  # Dynamic resizing to improve small-face detection
        faces = detect_faces_at_scales(frame, scales)

        for (x, y, x2, y2) in faces:
            # Crop the face region
            face = frame[y:y2, x:x2]

            # Skip if the face region is empty
            if face.size == 0:
                continue

            # Get embedding of the face
            face_embedding = get_face_embedding(face)

            # Check if the face is new
            if is_new_face(face_embedding, detected_faces_embeddings):
                detected_faces_embeddings.append(face_embedding)

                # Save the cropped face
                face_file_path = os.path.join(output_dir, f"face_{face_id}.jpg")
                cv2.imwrite(face_file_path, face)
                print(f"Saved: {face_file_path}")

                face_id += 1

    # Release the video capture
    cap.release()
    print("Processing complete. Faces saved in:", output_dir)

video_path = "/Users/noalevy/Desktop/GSCIL/homemade_face_extraction/video_manif.mp4"  # Path to your video file
output_dir = "faces_output"  # Directory to save the cropped face images
detect_faces_in_video(video_path, output_dir)
