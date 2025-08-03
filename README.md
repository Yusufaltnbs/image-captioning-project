Image Captioning with DenseNet201 and LSTM

This project is an image captioning system built for the OBSS Internship 2025 competition. It uses a DenseNet201 pretrained convolutional neural network as a feature extractor to extract image features, combined with an LSTM-based language model to generate natural language captions for images.

The project includes:

Data preprocessing and cleaning of captions.

Extraction of image features using DenseNet201.

Training of a caption generator model using image features and text sequences.

Implementation of beam search decoding (two versions) for more accurate caption generation.

Saving/loading of models and tokenizer.

Inference pipeline for generating captions on new images.

Batch caption generation for test images with saving results to CSV and Excel.

Project Structure
model.keras : Trained captioning model weights.

feature_extractor_obss.keras : DenseNet201 feature extractor model.

tokenizer_obss.pkl : Tokenizer fitted on captions vocabulary.

image_caption.csv / image_caption.xlsx : Generated captions on test images.

README.md : This file.

notebook.ipynb : Jupyter notebook with full training and inference code.

Drive links for large model files (model.keras and feature_extractor_obss.keras) due to size constraints.

Setup and Requirements
Python 3.x

TensorFlow 2.x

Keras

numpy, pandas, matplotlib, seaborn, tqdm

openpyxl (for Excel export)

You can install dependencies with:


pip install tensorflow numpy pandas matplotlib seaborn tqdm openpyxl
Usage
1. Load models and tokenizer

from tensorflow.keras.models import load_model
import pickle

caption_model = load_model("model.keras")
feature_extractor = load_model("feature_extractor_obss.keras")

with open("tokenizer_obss.pkl", "rb") as f:
    tokenizer = pickle.load(f)
Note: model.keras and feature_extractor_obss.keras files are large and available via this Drive link.

2. Generate caption for a new image

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def beam_search_caption_v2(model, tokenizer, photo_feature, max_length=31, beam_width=5, alpha=0.7):
    # Implementation of beam search with length normalization and repetition penalty (as in the project)
    pass  # Use the function from the code

image_path = "path/to/image.jpg"
img = load_img(image_path, target_size=(224, 224))
img = img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

feature = feature_extractor.predict(img)
caption = beam_search_caption_v2(caption_model, tokenizer, feature)
print("Generated Caption:", caption)
3. Batch caption generation and saving
Refer to the notebook for code on generating captions for multiple images and saving to CSV/Excel.

Model Details
Feature Extractor: DenseNet201 pretrained on ImageNet, outputting 1920-dimensional feature vectors.

Captioning Model:

Dense layer (300 units) on image features

Embedding layer initialized with GloVe 300d embeddings (non-trainable)

LSTM layer with dropout for sequence modeling

Fully connected layers with dropout and ReLU

Softmax output over vocabulary

Text processing:

Lowercasing, punctuation removal, removal of short words

Special tokens startseq and endseq used

Decoding: Beam search with length normalization and repetition penalty.

Dataset
Training images and captions are from OBSS Internship 2025 competition dataset.

Data split: 85% training, 15% validation by image IDs.


Contact
For any questions or collaboration, please contact:

Yusuf Altunbaş
 Computer Engineering 

Email: altunbasy1@gmail.com
GitHub: https://github.com/Yusufaltnbs
