import os
import time
import string
from PIL import Image
import numpy as np
from pickle import load,dump
import tensorflow as tf
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, get_file
from keras.layers import add
from keras.models import Model,load_model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.utils import plot_model

"""
Image Captioning Preprocessing Pipeline
---------------------------------------

This script performs the following steps:

1. Loads and cleans the Flickr8k captions.
2. Builds a vocabulary of cleaned words.
3. Saves the cleaned descriptions to file.
4. Downloads and uses a pre-trained Xception CNN model to extract image features.
5. Saves extracted image features for later use.
6. Tokenizes the cleaned captions for training.
7. Calculates maximum sequence length of captions for padding during training.


"""


# Loading a text file
def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text
# get all images with captions

def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    description = {}
    for caption in captions[:-1]:
        img, img_caption = caption.split('\t')
        if img[:-2] not in description:
            description[img[:-2]] = [caption]
        else:
            description[img[:-2]].append(caption)
    return description

# Data Cleaning - convert word in lower character, remove punctuation and words containing numbers
def clean_data(description):
    table = str.maketrans("", "", string.punctuation)
    for img,captions in description.items():
        for i,img_cap in enumerate(captions):
            img_cap = img_cap.replace('-', ' ')
            desc = img_cap.split()
            # convert them into lower characters
            desc = [word.lower() for word in desc]
            # remove punctuation
            desc = [word.translate(table) for word in desc]
            # remove hanging word such as 's and a
            desc = [word for word in desc if len(word)>1]
            # remove numbers
            desc = [word for word in desc if word.isalpha()]
            # convert back to string
            img_cap = " ".join(desc)
            description[img][i] = img_cap
    return description

# creating text_vocablary
def text_vocablary(description):
    vocab = set()
    for key in description.keys():
        [vocab.update(d.split()) for d in description[key]]
    return vocab

# Store the preprocessed descrption
def store_description(description, filename):
    lines = list()
    for key, desc_list in description.items():
        for desc in desc_list:
            lines.append(key + "\t" + desc)
    data = '\n'.join(lines)
    file = open(filename,'w')
    file.write(data)
    file.close()


image_dataset = "Flickr8k_Dataset/Flicker8k_Dataset"
text_dataset = "Flickr8k_text"

filename = text_dataset+"\Flickr8k.token.txt"

# creating dictnary for all image and there captions
despcriptions = all_img_captions(filename)

# cleaning data
processed_data = clean_data(despcriptions)

# Creating vocab
vocab = text_vocablary(processed_data)

# store data 
store_description(processed_data,"src\processed_description.txt")
print(len(vocab))

# Download the Weigths of tht model 
def download_weights(url, filename, cache_dir=None, max_retries=5):

    for attempt in range(max_retries):
        try:
            # Try to download the file
            return get_file(filename, origin=url, cache_dir=cache_dir)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)  # Wait before retrying
            else:
                raise
 
weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"

weights_path = download_weights(weights_url, "model_weights.h5", cache_dir="src")

model = Xception(include_top=False,pooling="avg",weights=weights_path)


# Feature Extraction (using my Xception model)

def feature_extraction(directory):
    features = {}
    valid_images = ['.jpg', '.jpeg', '.png']
    for img in tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = directory+"/"+img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image,axis=0)
        image = preprocess_input(image)
        # image = image/127.5
        # image = image - 1
        feature = model.predict(image)
        features[img] = feature
    
    return features

# features = feature_extraction(image_dataset)
# dump(features, open("src/features.p","wb"))

features = load(open('src/features.p', 'rb'))

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(image_dataset,photo))]
    return photos_present

def load_clean_description(filename,photos):
    file = load_doc(filename)
    despcriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words)<1:
            continue

        image, image_caption = words[0],words[1:]
        if image in photos:
            if image not in despcriptions:
                despcriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) +" <end>"
            despcriptions[image].append(desc)

    return despcriptions

def load_features(photos):
    all_features = load(open("src/features.p",'rb'))
    features = {k:all_features[k] for k in photos}
    return features

filename = text_dataset+'/'+"Flickr_8k.trainImages.txt"

train_imgs = load_photos(filename)
train_descriptions = load_clean_description("src\processed_description.txt", train_imgs)
train_features = load_features(train_imgs)

def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_token(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer  = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

tokenizer = create_token(train_descriptions)
dump(tokenizer, open('src/tokenizer.p','wb'))
vocab_size = len(tokenizer.word_index) + 1
print(f"this is the vocab size:{vocab_size}")

# calculating maximum length of descriptions
def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

max_length = max_length(train_descriptions)
print(f"this is the max_length:{ max_length}")



def data_generator(descriptions, features, tokenizer, max_length):
    def generator():
        while True:
            for key, description_list in descriptions.items():
                feature = features[key][0]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
                for i in range(len(input_image)):
                    yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]
    
    # Define the output signature for the generator
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    
    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    return dataset.batch(32)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

dataset = data_generator(train_descriptions, features, tokenizer, max_length)

for (a, b) in dataset.take(1):
    print(a['input_1'].shape, a['input_2'].shape, b.shape)
    break


def define_model(vocab_size, max_length):
    # CNN model from 2048 nodes to 256 nodes
    input1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(input1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM model
    input2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(input2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2,se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size,activation="softmax")(decoder2)

    # Tieing it togather [image,seq] [word]
    model = Model(inputs=[input1, input2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Model summary
    print(model.summary())
    plot_model(model, to_file = 'src.model.png', show_shapes=True)

    return model

# Training Model

model = define_model(vocab_size,max_length)
epochs = 10
steps_per_epoch = 5

# making a directory models to save our models
os.makedirs('models', exist_ok=True)
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)
    model.save(f'models/model_{i}.h5')
