import pickle
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open("C:\\Users\\Hp\\Downloads\\unique_data.pkl", "rb") as f:
    unique = pickle.load(f)

def get_sample(gender, age_group, district, data_dict):
    skey = gender + age_group + district
    if skey in data_dict:
        return random.choice(data_dict[skey])

    partial_key = gender + age_group
    partial_matches = [v for k, v in data_dict.items() if k.startswith(partial_key)]
    if partial_matches:
        return random.choice(random.choice(partial_matches))

    gender_matches = [v for k, v in data_dict.items() if k.startswith(gender)]
    if gender_matches:
        return random.choice(random.choice(gender_matches))

# Example isage
# gender_input = "Male"
# age_input = "60+"
# district_input = "Salem"

# sample_text, sample_audio = get_sample(gender_input, age_input, district_input, unique)

# print("Selected Text:", sample_text)
# print("Audio Shape:", sample_audio.shape)
