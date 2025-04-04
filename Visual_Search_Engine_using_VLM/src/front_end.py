import streamlit as st
import numpy as np
from PIL import Image
import time
from scipy.spatial.distance import cdist
import os
import torch
import torchvision
from torchvision import transforms

# Page Config
st.set_page_config(
    page_title="Visual Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title-text {
        font-family: 'Roboto', sans-serif;
        color: #1E3D59;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .stImage:hover {
        transform: scale(1.02);
    }
    .upload-section {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 2rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model and setup transforms
@st.cache_resource
def load_model():
    model = torchvision.models.resnet18(weights="DEFAULT")
    model.eval()
    return model

@st.cache_data
def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load Data
@st.cache_data
def read_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vectors_path = os.path.join(current_dir, "Vectors.npy")
        names_path = os.path.join(current_dir, "Names.npy")
        
        if not os.path.exists(vectors_path) or not os.path.exists(names_path):
            st.error(f"❌ Required data files not found in {current_dir}")
            return None, None
            
        Vectors = np.load(vectors_path)
        Names = np.load(names_path)
        return Vectors, Names
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

def extract_features(image, model, transform):
    """Extract features from an uploaded image using the model."""
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    handle = model.avgpool.register_forward_hook(get_activation("avgpool"))
    
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_tensor = transform(image)
        with torch.no_grad():
            _ = model(img_tensor.unsqueeze(0))
        features = activation["avgpool"].numpy().squeeze()
        handle.remove()
        return features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        handle.remove()
        return None

def main():
    # Load data
    Vectors, Names = read_data()
    if Vectors is None or Names is None:
        return
    
    # Load model and transforms
    model = load_model()
    transform = get_transforms()
    
    # Render sidebar
    st.sidebar.image(
        "https://images.unsplash.com/photo-1527430253228-e93688616381?w=500&auto=format",
        use_container_width=True
    )
    
    st.sidebar.markdown("""
    ## About
    This AI-powered Visual Search Engine helps you find visually similar images from a dataset. 
    Simply select an image or upload your own, and let the AI find the most relevant matches!
    
    ### Features
    - 🎯 Fast similarity search
    - 🤖 AI-powered matching
    - 📊 ResNet-18 feature extraction
    - 📤 Custom image upload
    
    ### Technologies
    - Python
    - PyTorch & Torchvision
    - Streamlit
    - NumPy & SciPy
    """)
    
    # Main content
    st.markdown("<h1 class='title-text'>🔍 Visual Search Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #6c757d; margin-bottom: 2rem;'>Discover visually similar images using AI-powered search</p>", unsafe_allow_html=True)
    
    # Image input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your own image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_container_width=True)
                st.session_state["uploaded_img"] = img
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        if st.button("🎲 Pick Random Image", use_container_width=True):
            random_img = Names[np.random.randint(len(Names))]
            st.session_state["disp_img"] = random_img
            img_path = os.path.join(os.path.dirname(__file__), "images", random_img)
            try:
                img = Image.open(img_path)
                st.image(img, caption="Selected Image", use_container_width=True)
                st.session_state["uploaded_img"] = img
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Find similar images
    if st.button("🔍 Find Similar Images", use_container_width=True):
        if "uploaded_img" not in st.session_state:
            st.warning("⚠️ Please select or upload an image first!")
            return
            
        with st.spinner("🔍 Finding similar images..."):
            # Extract features from the query image
            query_features = extract_features(st.session_state["uploaded_img"], model, transform)
            
            if query_features is not None:
                # Calculate distances and get top 5 similar images
                distances = cdist(query_features[None, ...], Vectors).squeeze()
                top5 = distances.argsort()[:5]
                
                # Display results
                st.markdown("""
                <h3 style='text-align: center; color: #1E3D59; margin: 2rem 0;'>
                    🎯 Top 5 Similar Images
                </h3>
                """, unsafe_allow_html=True)
                
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    img_path = os.path.join(os.path.dirname(__file__), "images", Names[top5[i]])
                    try:
                        img = Image.open(img_path)
                        col.image(img, use_container_width=True)
                    except Exception as e:
                        col.error(f"Error loading image: {str(e)}")
                
                st.success("✨ Similar images found successfully!")

if __name__ == "__main__":
    main()
