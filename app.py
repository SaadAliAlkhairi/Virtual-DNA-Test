import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import cv2
import eye_color
import os



# def skin_tone():
#     skin_tone_model = load_model('skin_tone_model.h5')
    
#     # List of class names
#     classes = ['Fair_Light', 'Medium_Tan', 'Dark_Deep']

#     # Mapping dictionary for descriptive skin tone labels
#     descriptive_labels = {
#         'Fair_Light': 'Fair / Light', # mother_image_path = '/home/saad/Documents/6th-Semester/AI/SemesterProject/Eye-Color-Detectionfinal/Eye-Color-Detection/mother4.jpg'
#     # father_image_path = '/home/saad/Documents/6th-Semester/AI/SemesterProject/Eye-Color-Detectionfinal/Eye-Color-Detection/father1.jpg'
#     # child_im
#         'Medium_Tan': 'Medium / Tan',
#         'Dark_Deep': 'Dark / Deep'
#     }

#     # Load the MTCNN face detection model
#     mtcnn = MTCNN()

#     uploaded_file = st.file_uploader('Upload an image ', type=['jpg', 'jpeg', 'png'])

#     # Display uploaded image
#     if uploaded_file is not None:
#         image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(image, 1)
        
        
#         # Predict button
#         if st.button('Predict Skin Tone'):
#             # Detect faces
#             try:
#                 faces = mtcnn.detect_faces(image)
#                 if len(faces) > 0:
#                     largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
#                     x, y, w, h = largest_face['box']
#                     detected_face = image[y:y+h, x:x+w]
                    
#                     # Resize the detected face to the desired input shape
#                     detected_face = cv2.resize(detected_face, (120, 90))
                    
#                     # Preprocess the detected face for classification
#                     detected_face = tf.keras.applications.mobilenet_v2.preprocess_input(detected_face[np.newaxis, ...])
                    
#                     # Predict the class of the face
#                     predictions = skin_tone_model.predict(detected_face)
#                     predicted_class_idx = np.argmax(predictions)
#                     predicted_class = classes[predicted_class_idx]
                    
#                     # Get the descriptive label from the mapping dictionary
#                     descriptive_label = descriptive_labels[predicted_class]
                    
#                     # Display the prediction with a larger font and a message
#                     st.write('')
#                     st.write('')
#                     st.write('')
#                     st.write('**Predicted Skin Tone:**')
#                     st.write(f'# {descriptive_label}')
#                 else:
#                     st.write('No face detected in the uploaded image.')
#             except Exception as e:
#                 st.write(f'Error detecting faces: {e}')









def main():

    blood_groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
    eye_colors = ['blue', 'brown', 'green']
    skin_tones = ['fair', 'light', 'tan']
    child_missing = True
    father_missing = True
    mother_missing = True

    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    st.title('VIRTUAL DNA TEST')

    with st.expander("Child Details"):
        child_face_image = st.file_uploader("Upload Child's FACE Image", type=["jpg", "png"])
        if child_face_image:
            # Save the uploaded image
            child_face_image_path = os.path.join(upload_dir, 'child_face_image.jpg')
            with open(child_face_image_path, 'wb') as f:
                f.write(child_face_image.read())
        # child_eye_image = st.file_uploader("Upload Child's EYE FOCUSED Image", type=["jpg", "png"])
        child_fingerprint_image = st.file_uploader("Upload Child's FINGERPRINT Image", type=["jpg", "png"])
        child_blood_group_index = st.selectbox("Select CHILD's Blood Group", options=range(len(blood_groups)), format_func=lambda x: blood_groups[x], index=0)
        child_blood_group = blood_groups[int(child_blood_group_index)]
        child_skin_tone_index = st.selectbox("Select CHILD's Skin Tone", options=range(len(skin_tones)), format_func=lambda x: skin_tones[x], index=0)
        child_skin_tone = skin_tones[int(child_skin_tone_index)]
        child_eye_color_index = st.selectbox("Select CHILD's Eye Color", options=range(len(eye_colors)), format_func=lambda x: eye_colors[x], index=0)
        child_eye_color = eye_colors[int(child_eye_color_index)]


    with st.expander("Father Details"):
        father_face_image = st.file_uploader("Upload Father's Face Image", type=["jpg", "png"])
        if father_face_image:
            # Save the uploaded image
            father_face_image_path = os.path.join(upload_dir, 'child_face_image.jpg')
            with open(father_face_image_path, 'wb') as f:
                f.write(father_face_image.read())
        # father_eye_image = st.file_uploader("Upload Father's EYE FOCUSED Image", type=["jpg", "png"])
        father_fingerprint_image = st.file_uploader("Upload Father's FINGERPRINT Image", type=["jpg", "png"])
        father_blood_group_index = st.selectbox("Select FATHER's Blood Group", options=range(len(blood_groups)), format_func=lambda x: blood_groups[x], index=0)
        father_blood_group = blood_groups[int(father_blood_group_index)]
        father_skin_tone_index = st.selectbox("Select FATHER's Skin Tone", options=range(len(skin_tones)), format_func=lambda x: skin_tones[x], index=0)
        father_skin_tone = skin_tones[int(father_skin_tone_index)]
        father_eye_color_index = st.selectbox("Select FATHER's Eye Color", options=range(len(eye_colors)), format_func=lambda x: eye_colors[x], index=0)
        father_eye_color = eye_colors[int(father_eye_color_index)]

    with st.expander("Mother Details"):
        mother_face_image = st.file_uploader("Upload Mother's Face Image", type=["jpg", "png"])
        if mother_face_image:
            # Save the uploaded image
            mother_face_image_path = os.path.join(upload_dir, 'child_face_image.jpg')
            with open(mother_face_image_path, 'wb') as f:
                f.write(mother_face_image.read())
        # mother_eye_image = st.file_uploader("Upload Mother's EYE FOCUSED Image", type=["jpg", "png"])
        mother_fingerprint_image = st.file_uploader("Upload Mother's FINGERPRINT Image", type=["jpg", "png"])
        mother_blood_group_index = st.selectbox("Select MOTHER's Blood Group", options=range(len(blood_groups)), format_func=lambda x: blood_groups[x], index=0)
        mother_blood_group = blood_groups[int(mother_blood_group_index)]
        mother_skin_tone_index = st.selectbox("Select MOTHER's Skin Tone", options=range(len(skin_tones)), format_func=lambda x: skin_tones[x], index=0)
        mother_skin_tone = skin_tones[int(mother_skin_tone_index)]
        mother_eye_color_index = st.selectbox("Select MOTHER's Eye Color", options=range(len(eye_colors)), format_func=lambda x: eye_colors[x], index=0)
        mother_eye_color = eye_colors[int(mother_eye_color_index)]






    if st.button('Submit'):
        if not(child_face_image and 
            #    child_eye_image and 
               child_fingerprint_image and child_blood_group and child_skin_tone and child_eye_color):
            st.write("CHILD's details missing")
        else:
            child_missing = False
        if not(father_face_image and 
            #    father_eye_image and 
               father_fingerprint_image and father_blood_group and father_skin_tone and father_eye_color):
            st.write("FATHER's details missing")
        else:
            father_missing = False
        if not(mother_face_image and 
            #    mother_eye_image and 
               mother_fingerprint_image and mother_blood_group and mother_skin_tone and mother_eye_color):
            st.write("MOTHER's details missing")
        else:
            mother_missing = False
        if not(child_missing) and not(father_missing) and not(mother_missing):
            eye_color_relatedness = eye_color.eye_color_detection(mother_face_image_path, father_face_image_path, child_face_image_path)
            # st.title('OUTPUT')
            # video_file = open('video.mp4', 'rb')
            # video_bytes = video_file.read()
            # st.video(video_bytes)
            st.title('OUTPUT')
            st.title(eye_color_relatedness)
            st.markdown(
                """
                <script>
                var video = document.getElementById("myVideo");
                video.scrollIntoView({behavior: "smooth", block: "end", inline: "nearest"});
                </script>
                """
            )
            video_file = open('video.mp4', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes, format="video/mp4", start_time=0)


if __name__ == "__main__":
    main()

