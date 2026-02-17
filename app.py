import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(page_title="Fruit Recognition", page_icon="🍎", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model_path = "fruit_model_best.pth" if os.path.exists("fruit_model_best.pth") else "fruit_model.pth"
    
    if not os.path.exists(model_path):
        st.error("Model file not found! Train the model first.")
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    num_classes = len(classes)
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, classes

st.title("🍎 Fruit Recognition AI")
st.markdown("Upload a fruit image for instant recognition")

with st.sidebar:
    st.header("ℹ️ Model Info")
    
    model, classes = load_model()
    
    if classes:
        st.metric("Device", str(device).upper())
        st.metric("Total Classes", len(classes))
        
        if os.path.exists("fruit_model_best.pth"):
            checkpoint = torch.load("fruit_model_best.pth", map_location=device)
            if "best_val_acc" in checkpoint:
                st.metric("Best Accuracy", f"{checkpoint['best_val_acc']:.1f}%")
        
        with st.expander("View all fruits"):
            for i, fruit in enumerate(classes, 1):
                st.write(f"{i}. {fruit}")

model, classes = load_model()

if model is None:
    st.stop()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🤖 Prediction")
        
        with st.spinner("Analyzing..."):
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                top5_prob, top5_idx = torch.topk(probabilities, min(5, len(classes)))
                
                predicted_idx = top5_idx[0].item()
                predicted_class = classes[predicted_idx]
                confidence = top5_prob[0].item() * 100
            
            if confidence > 70:
                st.success(f"🍊 **{predicted_class}**")
            elif confidence > 50:
                st.warning(f"🍊 **{predicted_class}**")
            else:
                st.info(f"🍊 **{predicted_class}** (low confidence)")
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            if confidence > 85:
                st.balloons()
    
    st.divider()
    
    st.subheader("📊 Top 5 Predictions")
    
    for i in range(min(5, len(classes))):
        idx = top5_idx[i].item()
        fruit_name = classes[idx]
        prob = top5_prob[i].item()
        
        col_a, col_b, col_c = st.columns([2, 3, 1])
        
        with col_a:
            if i == 0:
                st.markdown(f"**{i+1}. {fruit_name}**")
            else:
                st.write(f"{i+1}. {fruit_name}")
        
        with col_b:
            st.progress(float(prob))
        
        with col_c:
            st.write(f"{prob*100:.1f}%")
    
    with st.expander("🔍 Technical Details"):
        st.write(f"**Model:** ResNet18 (Transfer Learning)")
        st.write(f"**Device:** {device}")
        st.write(f"**Predicted Index:** {predicted_idx}")
        st.write(f"**Total Classes:** {len(classes)}")

else:
    st.info("👆 Upload a fruit image to get started!")
    
    if classes:
        st.markdown("### 🍇 Sample Fruits:")
        
        cols = st.columns(3)
        for i, fruit in enumerate(classes[:12]):
            cols[i % 3].write(f"• {fruit}")

st.divider()
st.caption("Built with PyTorch + Streamlit | ResNet18 Transfer Learning")