# import streamlit as st
# from PIL import Image
# import torch
# from torchvision import transforms
# from io import BytesIO

# # --------------------------
# # ⚙️ Streamlit Page Config
# # --------------------------
# st.set_page_config(page_title="CycleGAN Image Translator 🎨", layout="wide", page_icon="🎭")

# st.markdown("""
#     <style>
#     body {
#         background-color: #0E1117;
#         color: white;
#     }
#     .stButton>button {
#         background-color: #059bdd;
#         color: white;
#         border-radius: 10px;
#         padding: 0.5em 1em;
#         font-size: 1.1em;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.title("🎨 CycleGAN Image Translator")
# st.markdown("Convert between **Sketch ↔ Real Image** using your trained model.")

# # --------------------------
# # 🧠 Load Model
# # --------------------------
# @st.cache_resource
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = torch.load("cyclegan_model.pth", map_location=device)
#     model.eval()
#     return model, device

# model, device = load_model()

# # --------------------------
# # 🖼 Image Processing Utils
# # --------------------------
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# def tensor_to_image(tensor):
#     tensor = tensor.squeeze(0).detach().cpu()
#     tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
#     return transforms.ToPILImage()(tensor)

# # --------------------------
# # 🚀 UI Workflow
# # --------------------------
# uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     input_image = Image.open(uploaded_file).convert("RGB")
#     st.image(input_image, caption="Uploaded Image", use_container_width=True)

#     if st.button("✨ Generate"):
#         with st.spinner("Running the model... please wait ⏳"):
#             input_tensor = transform(input_image).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 output = model(input_tensor)
#             output_image = tensor_to_image(output)

#         st.image(output_image, caption="Generated Output", use_container_width=True)

#         # Option to download
#         buf = BytesIO()
#         output_image.save(buf, format="JPEG")
#         byte_im = buf.getvalue()
#         st.download_button("📥 Download Result", data=byte_im, file_name="output.jpg", mime="image/jpeg")

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import io

# Set page config
st.set_page_config(
    page_title="Face ↔ Sketch CycleGAN",
    page_icon="🎨",
    layout="wide"
)

# Generator Architecture (same as training)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels = in_channels * 2
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_channels)]
        
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, 
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels = in_channels // 2
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# Cache models to avoid reloading
@st.cache_resource
def load_models():
    """Load both generator models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Photo → Sketch model
    G_AB = Generator().to(device)
    checkpoint_ab = torch.load('photo_to_sketch.pth', map_location=device)
    G_AB.load_state_dict(checkpoint_ab['model_state_dict'])
    G_AB.eval()
    
    # Load Sketch → Photo model
    G_BA = Generator().to(device)
    checkpoint_ba = torch.load('sketch_to_photo.pth', map_location=device)
    G_BA.load_state_dict(checkpoint_ba['model_state_dict'])
    G_BA.eval()
    
    return G_AB, G_BA, device


def preprocess_image(image, target_size=256):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)


def postprocess_image(tensor):
    """Convert model output back to PIL Image"""
    image = tensor.cpu().squeeze().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 0.5 + 0.5).clip(0, 1)  # Denormalize
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def detect_image_type(image):
    """
    Simple heuristic to detect if image is a sketch or photo
    Sketches typically have higher contrast and less color variation
    """
    img_array = np.array(image.convert('L'))
    
    # Calculate statistics
    std_dev = np.std(img_array)
    mean_val = np.mean(img_array)
    
    # Sketches tend to have higher std deviation and be closer to extremes
    if std_dev > 80 and (mean_val > 180 or mean_val < 100):
        return "sketch"
    else:
        return "photo"


def convert_image(image, model, device):
    """Convert image using the specified model"""
    input_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    return postprocess_image(output_tensor)


# Main App
def main():
    st.title("🎨 Face ↔ Sketch CycleGAN")
    st.markdown("Convert photos to sketches and sketches to photos using CycleGAN")
    
    # Load models
    try:
        G_AB, G_BA, device = load_models()
        st.success(f"✅ Models loaded successfully! Using: {device}")
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    conversion_mode = st.sidebar.radio(
        "Conversion Mode",
        ["Auto-detect", "Photo → Sketch", "Sketch → Photo"],
        help="Auto-detect will automatically determine the input type"
    )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Input")
        upload_method = st.radio("Upload method:", ["Upload Image", "Use Camera"])
        
        if upload_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a photo or sketch"
            )
            
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
                st.image(input_image, caption="Input Image", use_container_width=True)
        else:
            camera_photo = st.camera_input("Take a picture")
            if camera_photo is not None:
                input_image = Image.open(camera_photo)
                st.image(input_image, caption="Captured Image", use_container_width=True)
            else:
                input_image = None
    
    with col2:
        st.header("📥 Output")
        
        if 'input_image' in locals() and input_image is not None:
            # Determine conversion direction
            if conversion_mode == "Auto-detect":
                detected_type = detect_image_type(input_image)
                st.info(f"🔍 Detected: {detected_type.upper()}")
                
                if detected_type == "photo":
                    output_image = convert_image(input_image, G_AB, device)
                    conversion_text = "Photo → Sketch"
                else:
                    output_image = convert_image(input_image, G_BA, device)
                    conversion_text = "Sketch → Photo"
            
            elif conversion_mode == "Photo → Sketch":
                output_image = convert_image(input_image, G_AB, device)
                conversion_text = "Photo → Sketch"
            
            else:  # Sketch → Photo
                output_image = convert_image(input_image, G_BA, device)
                conversion_text = "Sketch → Photo"
            
            st.image(output_image, caption=f"Output ({conversion_text})", use_container_width=True)
            
            # Download button
            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Download Result",
                data=byte_im,
                file_name=f"cyclegan_output_{conversion_text.replace(' → ', '_to_')}.png",
                mime="image/png"
            )
        else:
            st.info("👆 Upload or capture an image to see the conversion")
    
    # Information section
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        ### CycleGAN Face-Sketch Converter
        
        This application uses CycleGAN (Cycle-Consistent Generative Adversarial Networks) 
        to convert between face photos and sketches.
        
        **Features:**
        - 🎨 Photo to Sketch conversion
        - 🖼️ Sketch to Photo conversion
        - 🔍 Automatic input type detection
        - 📸 Camera support
        
        **How it works:**
        CycleGAN learns to translate images between two domains without paired examples.
        It uses cycle consistency loss to ensure the translation is meaningful.
        
        **Model Details:**
        - Architecture: ResNet-based Generator
        - Training: Unpaired face-sketch dataset
        - Image size: 256x256 pixels
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>Made with ❤️ using Streamlit and PyTorch</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()