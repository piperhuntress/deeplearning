import streamlit as st
from PIL import Image


# ------Function definitions------------#


# Function for uploading file
def upload_file():
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).resize((224, 224))
        st.image(uploaded_image, caption="Uploaded Image", width=700)
        return uploaded_file, uploaded_image
    else:
        return None, None


# ------End Function definitions------------#

# App Information
st.title("My First Streamlit App")
st.header("User Profile")
st.write("Project by: Jasmine")

app_desc = "The app displays the uploaded image along with its details (size and filename), and provides a summary button to view all profile information in one place. It's a simple, user-friendly demonstration of basic Streamlit components including text inputs, file uploaders, and column layouts."

st.write(app_desc)


# Layout with columns
col1, col2 = st.columns(2)

with col1:
    user_name = st.text_input("What is your name?")
with col2:
    fav_lang = st.selectbox(
        "Favorite Programming Language", ["Python", "JavaScript", "Java", "C++"]
    )

# Display values from variables
if user_name:
    st.write(f"Hello, {user_name}!")

st.write(f"Favorite language for this project: {fav_lang}")

# Image Upload
st.subheader("Image Upload")

# Call the upload_file function
uploaded_file, uploaded_image = upload_file()

# Show image information
if uploaded_file is not None:
    st.write(f"Image size: {uploaded_image.size} pixels")
    st.write(f"File name: {uploaded_file.name}")
else:
    st.info("No image uploaded yet. Upload an image.")

# Display summary information when button is clicked
if st.button("Show Summary"):
    st.success(
        f"""
    **Profile Summary:**
    - **Name:** {user_name if user_name else 'Not provided'}
    - **Favorite Language:** {fav_lang}
    - **Image Ready:** {'Yes' if uploaded_file is not None else 'No - upload an image!'}
      """
    )
