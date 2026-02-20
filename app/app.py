import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import sys
import os
from torchvision import transforms
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import build_model

# ── Page Config ────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Analyzer",
    page_icon="🫁",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .pneumonia-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = build_model(num_classes=2).to(device)
    
    # Try v2 model first, fallback to v1
    if os.path.exists('models/chest_xray_model_v2.pth'):
        model.load_state_dict(torch.load('models/chest_xray_model_v2.pth', 
                                        map_location=device))
        model_version = "v2 (Improved)"
    elif os.path.exists('models/chest_xray_model.pth'):
        model.load_state_dict(torch.load('models/chest_xray_model.pth',
                                        map_location=device))
        model_version = "v1"
    else:
        return None, "Model not found"
    
    model.eval()
    return model, model_version

# ── Grad-CAM Function ──────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # ResNet18 last conv layer
        target_layer = model.layer4[-1].conv2
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class):
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy()

# ── Prediction Function ────────────────────────────
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
    
    predicted_class = output.argmax(dim=1).item()
    confidence = probs[predicted_class].item()
    
    # Generate Grad-CAM
    grad_cam = GradCAM(model)
    heatmap = grad_cam.generate(input_tensor, predicted_class)
    
    return predicted_class, confidence, probs.cpu().numpy(), heatmap

# ── Create Heatmap Overlay ─────────────────────────
def create_heatmap_overlay(image, heatmap):
    img_array = np.array(image.convert('RGB'))
    
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# ── Main App ───────────────────────────────────────
def main():
    # Header
    st.markdown('<p class="main-header">🫁 Chest X-Ray Disease Analyzer</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Pneumonia Detection System</p>',
                unsafe_allow_html=True)
    
    # Load model
    model, model_version = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/lungs.png", width=100)
        st.title("About")
        st.info(f"""
        **Model Version:** {model_version}
        
        This AI system analyzes chest X-ray images to detect pneumonia.
        
        **How it works:**
        1. Upload a chest X-ray image
        2. AI analyzes the image
        3. Get instant diagnosis with confidence score
        4. View heatmap showing areas of concern
        
        **Accuracy:** ~87-90%
        
        **Technologies:**
        - PyTorch
        - ResNet18
        - Grad-CAM
        - Streamlit
        """)
        
        st.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. "
                  "Not intended for actual medical diagnosis.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear chest X-ray image in JPG or PNG format"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
            if st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the X-ray..."):
                    pred_class, confidence, probs, heatmap = predict_image(image, model)
                    
                    classes = ['NORMAL', 'PNEUMONIA']
                    diagnosis = classes[pred_class]
                    
                    # Store in session state
                    st.session_state.diagnosis = diagnosis
                    st.session_state.confidence = confidence
                    st.session_state.probs = probs
                    st.session_state.heatmap = heatmap
                    st.session_state.image = image
    
    with col2:
        if 'diagnosis' in st.session_state:
            st.subheader("📊 Analysis Results")
            
            diagnosis = st.session_state.diagnosis
            confidence = st.session_state.confidence
            probs = st.session_state.probs
            
            # Result box
            if diagnosis == "NORMAL":
                st.markdown(f"""
                <div class="result-box normal-box">
                    <h2 style="color: #28a745; margin:0;">✅ NORMAL</h2>
                    <p style="font-size: 1.2rem; margin:0.5rem 0;">
                        No signs of pneumonia detected
                    </p>
                    <p style="font-size: 1.5rem; font-weight: bold; margin:0;">
                        Confidence: {confidence*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box pneumonia-box">
                    <h2 style="color: #dc3545; margin:0;">⚠️ PNEUMONIA DETECTED</h2>
                    <p style="font-size: 1.2rem; margin:0.5rem 0;">
                        Potential pneumonia indicators found
                    </p>
                    <p style="font-size: 1.5rem; font-weight: bold; margin:0;">
                        Confidence: {confidence*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability bars
            st.subheader("Probability Breakdown")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Normal", f"{probs[0]*100:.1f}%")
                st.progress(float(probs[0]))
            
            with col_b:
                st.metric("Pneumonia", f"{probs[1]*100:.1f}%")
                st.progress(float(probs[1]))
            
            # Heatmap
            st.subheader("🔥 AI Attention Heatmap")
            st.caption("Red areas show where the AI focused its attention")
            
            overlay = create_heatmap_overlay(st.session_state.image, 
                                            st.session_state.heatmap)
            st.image(overlay, use_container_width=True)
            
            # Interpretation
            st.markdown("""
            <div class="warning-box">
                <strong>How to interpret the heatmap:</strong><br>
                🔴 <strong>Red/Yellow areas:</strong> High attention - AI considers these regions important<br>
                🔵 <strong>Blue/Purple areas:</strong> Low attention - AI considers these regions less relevant
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Upload an X-ray image and click 'Analyze' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using PyTorch, Streamlit & ResNet18</p>
        <p>Educational Project • Not for Clinical Use</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
