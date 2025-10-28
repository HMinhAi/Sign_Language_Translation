import cv2
import mediapipe as mp
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
# IMAGE_FILES = ["1.jpg"]
# with mp_hands.Hands(
#     static_image_mode=True,
#     max_num_hands=2,
#     min_detection_confidence=0.5) as hands:
#     for idx, file in enumerate(IMAGE_FILES):
#         # Read an image, flip it around y-axis for correct handedness output (see
#         # above).
#         start = time.time()
#         image = cv2.flip(cv2.imread(file), 1)
#         # Convert the BGR image to RGB before processing.
#         results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#         # Print handedness and draw hand landmarks on the image.
#         print('Handedness:', results.multi_handedness)
#         end = time.time()
#         print(f"Processing time for {file}: {end - start:.4f} seconds")
#         if not results.multi_hand_landmarks:
#             continue
#         image_height, image_width, _ = image.shape
#         annotated_image = image.copy()
#         for hand_landmarks in results.multi_hand_landmarks:
#             print('hand_landmarks:', hand_landmarks)
#             print(
#                 f'Index finger tip coordinates: (',
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#             )
#         mp_drawing.draw_landmarks(
#             annotated_image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
#         cv2.imwrite(
#             '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
#         # Draw hand world landmarks.
#         if not results.multi_hand_world_landmarks:
#             continue
#         for hand_world_landmarks in results.multi_hand_world_landmarks:
#             mp_drawing.plot_landmarks(
#                 hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# cap = cv2.VideoCapture(r"C:\Users\ming2\Documents\FPT_University\Semester 5\DPL302m\Project\Sign_Language_Translation\Data_Keypoints\test\Accept\0.npy")

# with mp_hands.Hands(
# 	model_complexity=0,
# 	min_detection_confidence=0.5,
# 	min_tracking_confidence=0.5) as hands:
# 	while cap.isOpened():
# 		success, image = cap.read()
# 		if not success:
# 			print("Ignoring empty camera frame.")
# 			# If loading a video, use 'break' instead of 'continue'.
# 			continue

# 		# To improve performance, optionally mark the image as not writeable to
# 		# pass by reference.
# 		start = time.time()
# 		image.flags.writeable = False
# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 		results = hands.process(image)

# 		end = time.time()

# 		# print(end - start)
# 		# Draw the hand annotations on the image.
# 		image.flags.writeable = True
# 		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# 		if results.multi_hand_landmarks:
# 			for hand_landmarks in results.multi_hand_landmarks:
# 				mp_drawing.draw_landmarks(
# 					image,
# 					hand_landmarks,
# 					mp_hands.HAND_CONNECTIONS,
# 					mp_drawing_styles.get_default_hand_landmarks_style(),
# 					mp_drawing_styles.get_default_hand_connections_style())
# 		# Flip the image horizontally for a selfie-view display.
# 		cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
# 		if cv2.getWindowProperty('MediaPipe Hands', cv2.WND_PROP_VISIBLE) < 1:
# 			break
# cap.release()

# For video file input:
# CHANGE THIS PATH TO YOUR VIDEO FILE
video_path = r"C:\Users\ming2\Documents\FPT_University\Semester 5\DPL302m\Project\Sign_Language_Translation\Data\test\Where\4.mp4"


# Or use camera (0 for default camera)
# video_path = 0

cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
	print(f"Error: Cannot open video file: {video_path}")
	print("Please check if the file exists and is a valid video format.")
	exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Video opened successfully!")

frame_number = 0

with mp_hands.Hands(
	model_complexity=0,
	min_detection_confidence=0.4,
	min_tracking_confidence=0.4) as hands:
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print(f"End of video at frame {frame_number}")
			break

		frame_number += 1
		print(f"Processing frame {frame_number}/{frame_count}", end='\r')

		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = hands.process(image)

		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				# Draw hand landmarks
				mp_drawing.draw_landmarks(
					image,
					hand_landmarks,
					mp_hands.HAND_CONNECTIONS,
					mp_drawing_styles.get_default_hand_landmarks_style(),
					mp_drawing_styles.get_default_hand_connections_style())
				
				# Draw keypoint labels
				h, w, c = image.shape
				for idx, landmark in enumerate(hand_landmarks.landmark):
					cx, cy = int(landmark.x * w), int(landmark.y * h)
					cv2.putText(image, str(idx), (cx, cy), 
							   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
		
		# Display frame number
		cv2.putText(image, f'Frame: {frame_number}/{frame_count}', (10, 30), 
				   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
		
		# Show the image
		cv2.imshow('MediaPipe Hands - Video', image)
		
		# Calculate delay based on video FPS (if FPS is 0 or invalid, use 30ms)
		delay = int(3000 / fps) if fps > 0 else 30
		
		# Wait for key press (ESC to exit)
		key = cv2.waitKey(delay) & 0xFF
		if key == 27:  # ESC key
			print("\nVideo playback stopped by user")
			break
		elif cv2.getWindowProperty('MediaPipe Hands - Video', cv2.WND_PROP_VISIBLE) < 1:
			print("\nWindow closed")
			break

print(f"\nTotal frames processed: {frame_number}")
cap.release()
cv2.destroyAllWindows()