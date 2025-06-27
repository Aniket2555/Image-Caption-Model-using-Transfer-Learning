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
│   ├── main.py                  # Main pipeline script
│   ├── features.p               # Extracted image features (generated)
│   ├── tokenizer.p              # Tokenizer object (generated)
│   ├── processed_description.txt# Cleaned captions (generated)
│   ├── test.py                  # (Optional) Test script
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
- **Run the main pipeline:**
  ```sh
  python src/main.py
  ```
  This will preprocess captions, extract features, tokenize text, and (optionally) train the model.

## Training
- The model is defined and trained in `src/main.py`.
- Model checkpoints are saved in the `models/` directory after each epoch.
- You can adjust training parameters (epochs, steps per epoch) in the script.

## Model Files
- `models/`: Contains saved `.h5` model files for each epoch.
- `src/features.p`: Pickled image features (generated, large file).
- `src/tokenizer.p`: Pickled tokenizer object (generated).
- `src/processed_description.txt`: Cleaned and processed captions.
- `src.model.png`: Visualization of the model architecture.

## Results & Visualization
- Model architecture is saved as `src.model.png`.
- Training progress and sample outputs can be added to this section as you experiment.

## Troubleshooting
- **TensorFlow or Keras import errors:** Ensure you are using Python 3.10 and have installed all dependencies in your virtual environment.
- **plot_model errors:** Install both `pydot` and the Graphviz system package, and ensure Graphviz is in your PATH.
- **Large files not tracked by git:** Datasets, models, and generated files are excluded via `.gitignore`.

## Credits
- Flickr8k dataset: [https://forms.illinois.edu/sec/1713398](https://forms.illinois.edu/sec/1713398)
- Keras, TensorFlow, and other open-source libraries.