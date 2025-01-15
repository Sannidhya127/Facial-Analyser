import cv2
from deepface import DeepFace
import dlib
import numpy as np
import matplotlib.pyplot as plt


def calculate_symmetry(landmarks):
    """
    Calculate facial symmetry using distances between mirrored facial key points.
    """
    left_points = np.array([
        landmarks[36],  # Left eye outer corner
        landmarks[39],  # Left eye inner corner
        landmarks[31],  # Left nose edge
        landmarks[48],  # Left mouth corner
    ])

    right_points = np.array([
        landmarks[45],  # Right eye outer corner
        landmarks[42],  # Right eye inner corner
        landmarks[35],  # Right nose edge
        landmarks[54],  # Right mouth corner
    ])

    # Flip right points horizontally for symmetry calculation
    midpoint = landmarks[27][0]  # Nose bridge midpoint (vertical symmetry axis)
    mirrored_right_points = np.copy(right_points)
    mirrored_right_points[:, 0] = 2 * midpoint - right_points[:, 0]

    # Calculate Euclidean distances between left and mirrored right points
    distances = np.linalg.norm(left_points - mirrored_right_points, axis=1)
    avg_distance = np.mean(distances)
    
    # Symmetry score: Inverse proportional to average deviation
    symmetry_score = max(0, 100 - avg_distance)
    return symmetry_score


def get_landmarks(image):
    """
    Detect facial landmarks using dlib's shape predictor.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure you download this

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    # Use the first detected face
    shape = predictor(gray, faces[0])
    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return landmarks


def estimate_skin_tone(image, landmarks):
    """
    Estimate the skin tone by analyzing the average color within the face region.
    """
    # Use the bounding box around the face to extract the region of interest (ROI)
    x_coords = [landmarks[i][0] for i in range(17, 27)]  # Jawline to nose bridge (approx)
    y_coords = [landmarks[i][1] for i in range(17, 27)]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Crop the face region from the image
    face_region = image[y_min:y_max, x_min:x_max]

    # Convert to HSV color space to estimate skin tone (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv_image[:, :, 0])

    # Define basic skin tone ranges in the Hue channel (approximate)
    if avg_hue < 15:
        skin_tone = "Light"
    elif 15 <= avg_hue < 35:
        skin_tone = "Medium"
    else:
        skin_tone = "Dark"

    return skin_tone


def classify_face_shape(landmarks):
    """
    Classify the face shape based on the landmarks' width-to-height ratio.
    """
    # Calculate the width and height of the face using key landmarks
    left_cheek = landmarks[0][0]  # Left side of the face
    right_cheek = landmarks[16][0]  # Right side of the face
    chin = landmarks[8][1]  # Chin (vertical)

    face_width = right_cheek - left_cheek
    face_height = chin - landmarks[19][1]  # Nose bridge to chin

    ratio = face_width / face_height

    # Classify face shape based on the width-to-height ratio
    if ratio < 1.2:
        face_shape = "Oval"
    elif 1.2 <= ratio < 1.5:
        face_shape = "Round"
    else:
        face_shape = "Square"

    return face_shape


def analyze_image(image_path):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Image not found or unable to read.")
            return

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the input image
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title("Input Image")
        plt.show()

        # Analyze using DeepFace (excluding race detection)
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False
        )

        # Handle output structure changes
        if isinstance(analysis, list):
            analysis = analysis[0]

        # Extract results
        age = analysis.get('age', 'N/A')
        gender = analysis.get('gender', 'N/A')
        dominant_emotion = analysis.get('dominant_emotion', 'N/A')

        print("\n--- Analysis Results ---")
        print(f"Age: {age}")
        print(f"Gender: {gender}")
        print(f"Dominant Emotion: {dominant_emotion}")

        # Get landmarks
        landmarks = get_landmarks(image)
        if landmarks is None:
            print("No face landmarks detected. Please try a different image.")
            return

        # Symmetry calculations
        symmetry_score = calculate_symmetry(landmarks)

        # Skin tone analysis
        skin_tone = estimate_skin_tone(image, landmarks)

        # Face shape analysis
        face_shape = classify_face_shape(landmarks)

        # Skin quality and emotion boost (for fun)
        emotion_boost = 10 if dominant_emotion.lower() == 'happy' else 0

        # Combine features into a final score
        looks_score = symmetry_score + emotion_boost
        looks_score = min(looks_score, 100)  # Cap at 100

        # Assign a label based on score
        if looks_score <= 25:
            looks_label = "Chut Magnet"
        elif 26 <= looks_score <= 50:
            looks_label = "Mediocre Monarch"
        elif 51 <= looks_score <= 75:
            looks_label = "Certified Handsome Devil"
        else:
            looks_label = "Godlike Specimen"

        # Display the results
        print("\n--- Good Looks Rater ---")
        print(f"Symmetry Score: {symmetry_score:.2f}/100")
        print(f"Emotion Boost: +{emotion_boost}")
        print(f"Final Handsomeness Score: {looks_score:.2f}/100")
        print(f"Skin Tone: {skin_tone}")
        print(f"Face Shape: {face_shape}")
        print(f"Label: {looks_label}")

    except Exception as e:
        print(f"An error occurred: {e}")


# Input: Path to the image
image_path = input("Enter the path to the image: ").strip()
analyze_image(image_path)
