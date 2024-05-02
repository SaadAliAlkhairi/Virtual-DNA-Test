import streamlit as st

def main():

    blood_groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']
    eye_colors = ['blue', 'brown', 'green']
    skin_tones = ['fair', 'light', 'tan']
    child_missing = True
    father_missing = True
    mother_missing = True

    st.title('VIRTUAL DNA TEST')

    with st.expander("Child Details"):
        child_face_image = st.file_uploader("Upload Child's FACE Image", type=["jpg", "png"])
        child_eye_image = st.file_uploader("Upload Child's EYE FOCUSED Image", type=["jpg", "png"])
        child_fingerprint_image = st.file_uploader("Upload Child's FINGERPRINT Image", type=["jpg", "png"])
        child_blood_group_index = st.selectbox("Select CHILD's Blood Group", options=range(len(blood_groups)), format_func=lambda x: blood_groups[x], index=0)
        child_blood_group = blood_groups[int(child_blood_group_index)]
        child_skin_tone_index = st.selectbox("Select CHILD's Skin Tone", options=range(len(skin_tones)), format_func=lambda x: skin_tones[x], index=0)
        child_skin_tone = skin_tones[int(child_skin_tone_index)]
        child_eye_color_index = st.selectbox("Select CHILD's Eye Color", options=range(len(eye_colors)), format_func=lambda x: eye_colors[x], index=0)
        child_eye_color = eye_colors[int(child_eye_color_index)]


    with st.expander("Father Details"):
        father_face_image = st.file_uploader("Upload Father's Face Image", type=["jpg", "png"])
        father_eye_image = st.file_uploader("Upload Father's EYE FOCUSED Image", type=["jpg", "png"])
        father_fingerprint_image = st.file_uploader("Upload Father's FINGERPRINT Image", type=["jpg", "png"])
        father_blood_group_index = st.selectbox("Select FATHER's Blood Group", options=range(len(blood_groups)), format_func=lambda x: blood_groups[x], index=0)
        father_blood_group = blood_groups[int(father_blood_group_index)]
        father_skin_tone_index = st.selectbox("Select FATHER's Skin Tone", options=range(len(skin_tones)), format_func=lambda x: skin_tones[x], index=0)
        father_skin_tone = skin_tones[int(father_skin_tone_index)]
        father_eye_color_index = st.selectbox("Select FATHER's Eye Color", options=range(len(eye_colors)), format_func=lambda x: eye_colors[x], index=0)
        father_eye_color = eye_colors[int(father_eye_color_index)]

    with st.expander("Mother Details"):
        mother_face_image = st.file_uploader("Upload Mother's Face Image", type=["jpg", "png"])
        mother_eye_image = st.file_uploader("Upload Mother's EYE FOCUSED Image", type=["jpg", "png"])
        mother_fingerprint_image = st.file_uploader("Upload Mother's FINGERPRINT Image", type=["jpg", "png"])
        mother_blood_group_index = st.selectbox("Select MOTHER's Blood Group", options=range(len(blood_groups)), format_func=lambda x: blood_groups[x], index=0)
        mother_blood_group = blood_groups[int(mother_blood_group_index)]
        mother_skin_tone_index = st.selectbox("Select MOTHER's Skin Tone", options=range(len(skin_tones)), format_func=lambda x: skin_tones[x], index=0)
        mother_skin_tone = skin_tones[int(mother_skin_tone_index)]
        mother_eye_color_index = st.selectbox("Select MOTHER's Eye Color", options=range(len(eye_colors)), format_func=lambda x: eye_colors[x], index=0)
        mother_eye_color = eye_colors[int(mother_eye_color_index)]


    if st.button('Submit'):
        if not(child_face_image and child_eye_image and child_fingerprint_image and child_blood_group and child_skin_tone and child_eye_color):
            st.write("CHILD's details missing")
        else:
            child_missing = False
        if not(father_face_image and father_eye_image and father_fingerprint_image and father_blood_group and father_skin_tone and father_eye_color):
            st.write("FATHER's details missing")
        else:
            father_missing = False
        if not(mother_face_image and mother_eye_image and mother_fingerprint_image and mother_blood_group and mother_skin_tone and mother_eye_color):
            st.write("MOTHER's details missing")
        else:
            mother_missing = False
        if not(child_missing) and not(father_missing) and not(mother_missing):
            # st.title('OUTPUT')
            # video_file = open('video.mp4', 'rb')
            # video_bytes = video_file.read()
            # st.video(video_bytes)
            st.title('OUTPUT')
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

