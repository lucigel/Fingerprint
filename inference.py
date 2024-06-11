import torch 
import torchvision.transforms as transforms
import streamlit as st 
from PIL import Image

from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


img_size = 96
choose = ["ID", "GENDER", "HAND", "FINGER"]
input_shape = (img_size, img_size)
model = Model(input_shape, choose).to(device)

# Load the best model
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Điều chỉnh các giá trị này phù hợp với chuẩn hóa đã dùng khi huấn luyện
])

st.title("Fingerprints Classification")

uploaded_file = st.file_uploader("Choose an image..", type="bmp")

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        
        _, predicted_ID = torch.max(output["ID"], 1)
        gender_num = torch.round(torch.sigmoid(output["GENDER"])).item()
        hand_num = torch.round(torch.sigmoid(output["HAND"])).item()
        _, predicted_finger = torch.max(output["FINGER"], 1)
        
    
        gender = 'Male' if gender_num == 0 else 'Female'
        hand = 'Left' if hand_num == 0 else 'Right'
        finger_map = {0: 'Little', 1: 'Ring', 2: 'Middle', 3: 'Index', 4: 'Thumb'}
        finger = finger_map[predicted_finger.item()]
       
        st.write(f"Predicted ID: {predicted_ID.item()}")
        st.write(f"Predicted Gender: {gender}")
        st.write(f"Predicted Hand: {hand}")
        st.write(f"Predicted Finger: {finger}")