import nltk
import os

# Define the directory for NLTK data
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')

# Create the directory if it doesn't exist
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK packages
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
except Exception as e:
    print(f"Error downloading punkt: {e}")

try:
    nltk.download('wordnet', download_dir=nltk_data_dir)
except Exception as e:
    print(f"Error downloading wordnet: {e}")

try:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
except Exception as e:
    print(f"Error downloading punkt_tab: {e}")

print(f"NLTK data downloaded to: {nltk_data_dir}")
