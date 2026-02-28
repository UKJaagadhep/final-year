import torch
import re

def clean_tamil_text(text):
    # Keep Tamil unicode block + spaces
    text = re.sub(r'[^\u0B80-\u0BFF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Encode text into integer IDs ---
def encode_text(text, char2id):
    text = clean_tamil_text(text)
    return torch.tensor([char2id[c] for c in text if c in char2id], dtype=torch.long)

# --- Encode metadata (gender, age, district) ---
def encode_meta(input_meta):

    return torch.tensor([input_meta["gender"]], dtype=torch.long)

# --- Prepare inference batch ---
def prepare_inference_input(text, metadata, device='cpu'):
    char2id = {' ': 1, 'ஃ': 2, 'அ': 3, 'ஆ': 4, 'இ': 5, 'ஈ': 6, 'உ': 7, 'ஊ': 8, 'எ': 9,
               'ஏ': 10, 'ஐ': 11, 'ஒ': 12, 'ஓ': 13, 'க': 14, 'ங': 15, 'ச': 16, 'ஜ': 17, 'ஞ': 18,
               'ட': 19, 'ண': 20, 'த': 21, 'ந': 22, 'ன': 23, 'ப': 24, 'ம': 25, 'ய': 26, 'ர': 27,
               'ற': 28, 'ல': 29, 'ள': 30, 'ழ': 31, 'வ': 32, 'ஷ': 33, 'ஸ': 34, 'ஹ': 35, 'ா': 36,
               'ி': 37, 'ீ': 38, 'ு': 39, 'ூ': 40, 'ெ': 41, 'ே': 42, 'ை': 43, 'ொ': 44, 'ோ': 45,
               'ௌ': 46, '்': 47}
    gender2id= {'Female': 0, 'Male': 1}
    age2id = {'18-30': 0, '30-45': 1, '45-60': 2, '60+': 3}
    district2id = {'Ariyalur': 0, 'Coimbatore': 1, 'Cuddalore': 2, 'Dharmapuri': 3, 'Erode': 4, 'Kallakurichi': 5, 'Krishnagiri': 6, 'Mayiladuthurai': 7, 'Nagapattinam': 8, 'Namakkal': 9, 'Perambalur': 10, 'Pudukkottai': 11, 'Salem': 12, 'Sivaganga': 13, 'Thanjavur': 14, 'Tiruchirappalli': 15, 'Tiruppur': 16, 'Tiruvarur': 17, 'Viluppuram': 18}
    text_tensor = encode_text(text, char2id).unsqueeze(0).to(device)  # Add batch dim
    text_mask = (text_tensor != 0).long()
    meta_tensor = encode_meta(metadata).unsqueeze(0).to(device)
    
    return text_tensor, text_mask, meta_tensor


