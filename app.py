import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="AI Nutrient Deficiency Detector", layout="wide")

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh.FaceMesh()

# ---------- Helper Functions ----------

def detect_face(image):
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = image.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            return image[max(0,y1):y2, max(0,x1):x2]
    return None


def extract_face_regions(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    results = mp_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None, None, None

    pts = results.multi_face_landmarks[0]

    def get_landmark_point(idx):
        return int(pts.landmark[idx].x * w), int(pts.landmark[idx].y * h)

    # Eye crop
    eye_left = get_landmark_point(33)
    eye_right = get_landmark_point(263)
    y_min = min(eye_left[1], eye_right[1]) - 30
    y_max = max(eye_left[1], eye_right[1]) + 30
    x_min = eye_left[0] - 30
    x_max = eye_right[0] + 30
    eye_img = image[max(0,y_min):y_max, max(0,x_min):x_max]

    # Lip crop
    lip_top = get_landmark_point(13)
    lip_bottom = get_landmark_point(14)
    lip_left = get_landmark_point(78)
    lip_right = get_landmark_point(308)
    lip_img = image[lip_top[1]-20:lip_bottom[1]+20, lip_left[0]-20:lip_right[0]+20]

    # Cheek crop
    cheek_left = get_landmark_point(93)
    cheek_right = get_landmark_point(323)
    cheek_img = image[cheek_left[1]-40:cheek_left[1]+40, cheek_left[0]-40:cheek_left[0]+40]

    return eye_img, lip_img, cheek_img


# ---------- Prediction Logic ----------

def analyze_eye(eye):
    if eye is None or eye.size == 0:
        return "Not Detectable"

    hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:,:,2])

    if brightness < 80:
        return "Possible Iron Deficiency (pale eyes)"
    return "Normal"


def analyze_lips(lips):
    if lips is None or lips.size == 0:
        return "Not Detectable"

    hsv = cv2.cvtColor(lips, cv2.COLOR_BGR2HSV)
    red_channel = np.mean(lips[:,:,2])

    if red_channel < 120:
        return "Possible Vitamin B12 Deficiency (pale lips)"
    return "Normal"


def analyze_skin(cheek):
    if cheek is None or cheek.size == 0:
        return "Not Detectable"

    gray = cv2.cvtColor(cheek, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F).var()

    if lap < 40:
        return "Possible Dehydration (dry skin texture)"
    return "Normal"


# ---------- STREAMLIT UI ----------
st.title("🌟 AI-Based Nutrient Deficiency Detector")
st.write("Upload a face image. The system will analyze eyes, lips & skin.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    face = detect_face(img)

    if face is None:
        st.error("No face detected. Try another image.")
    else:
        st.image(face, caption="Detected Face", width=300)

        eye, lips, cheek = extract_face_regions(face)

        col1, col2, col3 = st.columns(3)
        with col1: 
            if eye is not None: st.image(eye, caption="Eyes")
        with col2:
            if lips is not None: st.image(lips, caption="Lips")
        with col3:
            if cheek is not None: st.image(cheek, caption="Skin / Cheek")

        st.subheader("🔍 Nutrient Deficiency Report")

        st.write("**Eyes:** ", analyze_eye(eye))
        st.write("**Lips:** ", analyze_lips(lips))
        st.write("**Skin:** ", analyze_skin(cheek))

        st.success("Analysis completed!")
