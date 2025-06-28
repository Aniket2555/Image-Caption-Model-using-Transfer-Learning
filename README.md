# Image Caption Model Project

## Overview
This project implements an end-to-end image captioning pipeline using the Flickr8k dataset. It covers data preprocessing, feature extraction with a pre-trained CNN (Xception), text tokenization, and training a neural network to generate captions for images.

## Dataset
- **Flickr8k_Dataset/**: Contains the images used for training and evaluation.
- **Flickr8k_text/**: Contains caption files and image splits (train, test, dev).

## Project Structure
```
image-caption-model-project-2/
├── src/
│   ├── main.py                  # Main pipeline script (preprocessing + training)
│   ├── test.py                  # Test script for single image captioning
│   ├── evaluate.py              # Evaluation script with BLEU scores
│   ├── features.p               # Extracted image features (generated)
│   ├── tokenizer.p              # Tokenizer object (generated)
│   ├── processed_description.txt# Cleaned captions (generated)
│   └── datasets/                # (Optional) Additional datasets
├── models/                      # Saved model checkpoints (generated)
├── research/
│   └── notebook.ipynb           # Research and experimentation notebook
├── Flickr8k_Dataset/            # Image dataset (not tracked by git)
├── Flickr8k_text/               # Caption and split files (not tracked by git)
├── .venv/                       # Virtual environment (not tracked by git)
├── .gitignore                   # Git ignore rules
├── pyproject.toml               # Project metadata
├── requirments.txt              # Python dependencies
├── uv.lock                      # Lock file for uv package manager
├── .python-version              # Python version file
├── src.model.png                # Model architecture visualization (generated)
└── README.md                    # Project documentation
```

## Setup Instructions
1. **Clone the repository**
   ```sh
   git clone <repo-url>
   cd image-caption-model-project-2
   ```
2. **Install Python 3.10** (if not already installed)
3. **Create and activate a virtual environment**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On Linux/Mac
   ```
4. **Install dependencies**
   ```sh
   uv pip install -r requirments.txt
   # or
   pip install -r requirments.txt
   ```
5. **Download the Flickr8k dataset and place it in the correct folders**
   - Images: `Flickr8k_Dataset/Flicker8k_Dataset/`
   - Captions: `Flickr8k_text/`

6. **(Optional) Install Graphviz for model visualization**
   - Download from https://graphviz.gitlab.io/download/ and add to PATH.
   - Install Python package: `uv pip install pydot` or `pip install pydot`

## Usage

### 1. **Run the main pipeline (preprocessing + training):**
  ```sh
  python src/main.py
  ```
  This will:
  - Preprocess and clean the captions
  - Extract image features using Xception CNN
  - Tokenize the text data
  - Train the image captioning model
  - Save model checkpoints after each epoch

### 2. **Test single image captioning:**
  ```sh
  python src/test.py --image path/to/your/image.jpg
  ```
  Example:
  ```sh
  python src/test.py --image Flickr8k_Dataset/Flicker8k_Dataset/17273391_55cfc7d3d4.jpg
  ```
  This will:
  - Load the trained model
  - Generate a caption for the specified image
  - Display the image with the generated caption

### 3. **Evaluate model performance:**
  ```sh
  python src/evaluate.py
  ```
  This will:
  - Load the test dataset
  - Generate captions for all test images
  - Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores
  - Display sample predictions vs references

## Model Architecture
The image captioning model consists of:
- **Encoder**: Xception CNN (pre-trained) for image feature extraction
- **Decoder**: LSTM with attention mechanism for caption generation
- **Input**: 2048-dimensional image features + text sequence
- **Output**: Probability distribution over vocabulary words

## Training
- The model is defined and trained in `src/main.py`
- Model checkpoints are saved in the `models/` directory after each epoch
- Training parameters can be adjusted in the script:
  - `epochs`: Number of training epochs (default: 10)
  - `batch_size`: Batch size for training (default: 32)
  - `max_length`: Maximum caption length (calculated from data)

## Model Files
- `models/`: Contains saved `.h5` model files for each epoch
- `src/features.p`: Pickled image features (generated, large file)
- `src/tokenizer.p`: Pickled tokenizer object (generated)
- `src/processed_description.txt`: Cleaned and processed captions
- `src.model.png`: Visualization of the model architecture

## Evaluation Metrics
The model is evaluated using BLEU scores:
- **BLEU-1**: Unigram precision
- **BLEU-2**: Bigram precision  
- **BLEU-3**: Trigram precision
- **BLEU-4**: 4-gram precision

## Results & Visualization
- Model architecture is saved as `src.model.png`
- Training progress is displayed during training
- Sample predictions are shown during evaluation
- Individual image testing displays the image with generated caption

## Troubleshooting

### Common Issues:
- **TensorFlow or Keras import errors:** Ensure you are using Python 3.10 and have installed all dependencies in your virtual environment
- **plot_model errors:** Install both `pydot` and the Graphviz system package, and ensure Graphviz is in your PATH
- **Large files not tracked by git:** Datasets, models, and generated files are excluded via `.gitignore`
- **Memory issues:** The features.p file can be large; ensure sufficient RAM for loading
- **CUDA/GPU issues:** The model can run on CPU, but GPU acceleration is recommended for faster training

### File Path Issues:
- Ensure image paths in test.py are correct relative to the project root
- Check that all required files (tokenizer.p, model files) exist in the expected locations

## Dependencies
Key dependencies include:
- TensorFlow/Keras
- PIL (Pillow)
- NumPy
- Matplotlib
- NLTK (for BLEU scoring)
- tqdm (for progress bars)

## Credits
- Flickr8k dataset: [https://forms.illinois.edu/sec/1713398](https://forms.illinois.edu/sec/1713398)
- Keras, TensorFlow, and other open-source libraries
- Xception CNN architecture for feature extraction