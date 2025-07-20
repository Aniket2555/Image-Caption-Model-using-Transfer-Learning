import os
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pickle import load
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu

# Load tokenizer
with open("src/tokenizer.p", "rb") as f:
    tokenizer = load(f)
vocab_size = len(tokenizer.word_index) + 1
max_length = 32  # Set to 32 as in test.py

# Define model architecture (same as in test.py)
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Load model weights
model = define_model(vocab_size, max_length)
model.load_weights('models/model_9.h5')

# Load Xception model for feature extraction
xception_model = Xception(include_top=False, pooling="avg")

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print(f"ERROR: Couldn't open image {filename}")
        return None
    image = image.resize((299,299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature

def generate_caption(model, tokenizer, photo_feature, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

# Load test image list
with open('Flickr8k_text/Flickr_8k.testImages.txt') as f:
    test_images = f.read().strip().split('\n')

# Load reference captions
references = {}
with open('src/processed_description.txt') as f:
    for line in f:
        tokens = line.strip().split('\t')
        if len(tokens) != 2:
            continue
        img, caption = tokens
        if img in test_images:
            if img not in references:
                references[img] = []
            # Add <start> and <end> tokens for consistency
            references[img].append(f'start {caption} end')

def get_image_path(img):
    return os.path.join('Flickr8k_Dataset/Flicker8k_Dataset', img)

# Generate captions and collect references
actual, predicted = [], []
for img in test_images:
    img_path = get_image_path(img)
    feature = extract_features(img_path, xception_model)
    if feature is None:
        continue
    yhat = generate_caption(model, tokenizer, feature, max_length)
    # Remove start/end tokens for BLEU
    yhat_clean = yhat.replace('start', '').replace('end', '').strip().split()
    refs = [ref.replace('start', '').replace('end', '').strip().split() for ref in references.get(img, [])]
    if refs:
        actual.append(refs)
        predicted.append(yhat_clean)

# Calculate BLEU scores
bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

print('BLEU-1: %f' % bleu1)
print('BLEU-2: %f' % bleu2)
print('BLEU-3: %f' % bleu3)
print('BLEU-4: %f' % bleu4)

# Save results to results/bleu_scores.txt
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, 'bleu_scores.txt')
with open(results_path, 'w') as f:
    f.write(f'BLEU-1: {bleu1:.6f}\n')
    f.write(f'BLEU-2: {bleu2:.6f}\n')
    f.write(f'BLEU-3: {bleu3:.6f}\n')
    f.write(f'BLEU-4: {bleu4:.6f}\n')
print(f'BLEU scores saved to {results_path}') 