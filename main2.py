### IMPORT STATEMENTS ###

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import string
from models import *
from configs import cfg
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


### HELPER FUNCTIONS ###

# returns a pandas dataframe of the given file
# fname: string; path to the file
# num_rows: int; number of rows to read (for partial datasets)
# return: pandas Dataframe
def load_data(fname, num_rows = None):
    return pd.read_csv(fname, nrows = num_rows)


# creates a one-hot encoding matrix for the given data
# data: 1d numpy array; list of items/features to encode
# dictionary: dict; mapping from the item to its index
# return: 2d numpy array
def encode_one_hot(data, dictionary):
    # creates the 2d array of zeros
    one_hot_encoding = np.zeros((data.shape[0], len(dictionary)))
    
    # gathers the respective indices of each item
    data_indices = [dictionary[item] for item in data]

    # encodes the 1 for all the items
    one_hot_encoding[range(data.shape[0]), data_indices] = 1
    
    return one_hot_encoding


# converts a one-hot encoding of the reviews into strings
# data: 3d torch list; list of one-hot encoding
# dictionary: dict; mapping from the index to the character
# return: review strings (1d list)
def decode_one_hot_reviews(data):
    extended_char = cfg['valid_char'] + 'SEP'
    decoded = [''.join([extended_char[torch.argmax(c)] for c in review]) for review in data]
    #decoded = [''.join([cfg['valid_char'][torch.argmax(c)] 
                #if torch.argmax(c) < cfg['valid_char_len'] 
                #else '' for c in review]) for review in data]
    
    return decoded


# cleans and processes (feature encoding) the training data
# orig_data: pandas Dataframe; raw data that is outputted from load_data
# returns: features (2d numpy array; one-hot), labels (1d numpy array of strings), beer dictionary (dict)
def process_train_data(orig_data):
    print ("Processing training data")
    print ("Original data shape: " + str(orig_data.shape))
    
    # takes the relevant columns
    data = orig_data[['beer/style', 'review/overall', 'review/text']].copy()
    
    # --- DATA CLEANING ---
    
    # drops the rows with missing data
    data.replace('', np.nan, inplace = True)
    data.dropna(inplace = True)
    data.reset_index(drop = True, inplace = True)

    # sets all characters to lower case
    data['beer/style'] = data['beer/style'].str.lower()
    data['review/text'] = data['review/text'].str.lower()

    # converts all whitespace (space, tabs, newlines, etc.) into spaces
    whitespace_regex = '[' + string.whitespace + ']'
    data['review/text'] = data['review/text'].str.replace(whitespace_regex, ' ', regex = True)

    # removes all invalid characters
    invalid_char_regex = '[^' + cfg['valid_char'] + ']'
    data['review/text'] = data['review/text'].str.replace(invalid_char_regex, '', regex = True)
    
    print ("Data shape after cleaning: " + str(data.shape))
    
    # --- DATA PROCESSING ---
    
    # creates a list of beer and a dictionary to map a beer style to an index
    beer_list = data['beer/style'].unique()
    beer_to_index = dict(zip(beer_list, range(beer_list.shape[0])))
    print ("Number of unique beers: " + str(beer_list.shape[0]))

    # creates the input features
    beer_encoding = encode_one_hot(data['beer/style'].values, beer_to_index)
    score_encoding = data['review/overall'].values
    score_encoding = score_encoding.reshape(score_encoding.shape[0], 1)
    input_features = np.hstack((beer_encoding, score_encoding))
    print ("Input feature matrix shape: " + str(input_features.shape))
    
    # creates the labels
    labels = data['review/text'].values
    print ("Labels matrix shape: " + str(labels.shape))
    
    return input_features, labels, beer_to_index


# updates the configurations based on the results of the processed dataset
def update_configurations(feature_length):
    # sets the models' input dimensions to the size of features (beer style + score) + character encoding
    cfg['input_dim'] = feature_length + cfg['output_dim']

    
# splits the dataset + labels into a training and validation set
# features: numpy array
# labels: numpy array
# percent_training: float; percentage (from 0.0 to 1.0) of data to be used for training
# returns: training features, training labels, validation features, validation labels (all numpy arrays)
def train_valid_split(features, labels, percent_training):
    # gets the index of where to split
    training_last_index = int(percent_training * features.shape[0])

    x_train = features[:training_last_index]
    y_train = labels[:training_last_index]

    x_valid = features[training_last_index:]
    y_valid = labels[training_last_index:]
    
    print ("Training set size: " + str(x_train.shape[0]))
    print ("Validation set size: " + str(x_valid.shape[0]))
    
    return x_train, y_train, x_valid, y_valid
    

# cleans and processes (feature encoding) the testing data
# orig_data: pandas Dataframe; raw data that is outputted from load_data
# dictionary: dict; mapping from the beer style to its index (output of process_train_data)
# returns: features (2d numpy array; one-hot)
def process_test_data(orig_data, dictionary):  
    print ("Processing the testing data")
    print ("Original data shape: " + str(orig_data.shape))
    
    # takes the relevant columns
    data = orig_data[['beer/style', 'review/overall']].copy()
    
    # --- DATA CLEANING ---
    
    # sets all characters to lower case
    data['beer/style'] = data['beer/style'].str.lower()
    
    # --- DATA PROCESSING ---
    
    # creates the input features
    beer_encoding = encode_one_hot(data['beer/style'].values, dictionary)
    score_encoding = data['review/overall'].values
    score_encoding = score_encoding.reshape(score_encoding.shape[0], 1)
    input_features = np.hstack((beer_encoding, score_encoding))
    print ("Input feature matrix shape: " + str(input_features.shape))
        
    return input_features


# pads the reviews so that all reviews in the set have an equal size
# and adds the <SOS> and <EOS> tags to the beginning and end of the reviews
# orig_data: 2d list of ints; list of reviews with the characters converted to their respective indices
# outputs: 2d numpy array of ints; padded reviews with the characters as indices
def pad_data(orig_data):
    # defines the character indices for the <SOS>, <EOS>, and <PAD> tags
    sos_tag_index = cfg['valid_char_len']
    eos_tag_index = sos_tag_index + 1
    pad_tag_index = eos_tag_index + 1
    
    # finds the longest review length
    review_lengths = [len(review) for review in orig_data]
    longest_review_length = np.max(review_lengths)
    
    # pads the reviews and adds the <SOS> and <EOS> tags
    padded_reviews = []
    for review in orig_data:
        pad_length = longest_review_length - len(review)
        padded_review = [sos_tag_index] + review + [eos_tag_index] + [pad_tag_index] * pad_length
        padded_reviews.append(padded_review)
        
    return np.array(padded_reviews)


def train(model, model_name, criterion, optimizer, computing_device, x_train, y_train, x_valid, y_valid, cfg):
    train_loss = []
    valid_loss = []
    valid_bleu = []

    start_time = time.time()
    
    softmax = nn.LogSoftmax(dim = 1)
    bleu_smoothing = SmoothingFunction()
    
    early_stop_count = 0
    min_loss = 100

    for epoch in range(1, cfg['epochs'] + 1):

        print ('----- Epoch #' + str(epoch) + ' -----')

        start_index = 0
        end_index = cfg['batch_size']

        losses = []

        print ('----- Training -----')
        while start_index < len(x_train):
            # takes the minibatch subset
            batch_x = x_train[start_index:end_index]
            batch_y = y_train[start_index:end_index]

            # converts the reviews char -> index
            indexed_reviews = [[char_to_index[c] for c in review] for review in batch_y]

            # pads the reviews
            padded_reviews = pad_data(indexed_reviews)

            # converts the review to a one-hot encoding
            # and concatenates this to the input features
            one_hot_length = cfg['output_dim']
            final_batch_x = []
            for features, reviews in zip(batch_x, padded_reviews):
                for char_index in reviews[:-1]:
                    one_hot_encoding = np.zeros(one_hot_length)
                    one_hot_encoding[char_index] = 1
                    final_features = np.hstack((features, one_hot_encoding))
                    final_batch_x.append(final_features)

            # converts the final array into a numpy array
            final_batch_x = np.array(final_batch_x)

            # resizes the flattened array into batch_size x sequence_length x feature_length
            final_batch_x.resize(padded_reviews.shape[0], padded_reviews.shape[1] - 1, final_batch_x.shape[1])

            # converts final input array to tensor
            final_batch_x = torch.from_numpy(final_batch_x).float().to(computing_device)

            # zeros the gradients
            optimizer.zero_grad()

            # passes the final input array to the model's forward pass
            outputs, _ = model(final_batch_x)
            soft_outputs = softmax(outputs)

            # prints the actual reviews vs the predicted reviews
            actual_reviews = batch_y
            predicted_reviews = decode_one_hot_reviews(soft_outputs)
            
            for i in range(1):
                print ("Actual Review: " + actual_reviews[i])
                print ("Predicted Review: " + predicted_reviews[i])
            
            # reshapes the outputs to N x feature_length (for the loss function)
            outputs = outputs.view(-1, outputs.shape[2])

            # creates the targets and reshapes it to a single dimension
            targets = torch.from_numpy(padded_reviews[:, 1:]).long()
            targets = targets.view(-1)

            # passes the outputs and targets to the loss function and backpropagates
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            print("Batch start index: " + str(start_index) + " | Loss: " + str(loss.item()))
            print("Time elapsed: " + str(time.time() - start_time))

            start_index = end_index
            end_index += cfg['batch_size']
            
            # stops training when the remaining data count is less than a minibatch
            if end_index > len(x_train):
                break


        train_loss.append(np.mean(losses))
        torch.save(model, model_name + "_e" + str(epoch) + ".pt")
        print()

        print ('----- Validating -----')
        start_index = 0
        end_index = cfg['batch_size']

        losses = []
        bleus = []

        with torch.no_grad():

            while start_index < len(x_valid):
                # takes the minibatch subset
                batch_x = x_valid[start_index:end_index]
                batch_y = y_valid[start_index:end_index]

                # converts the reviews char -> index
                indexed_reviews = [[char_to_index[c] for c in review] for review in batch_y]

                # pads the reviews
                padded_reviews = pad_data(indexed_reviews)

                # converts the review to a one-hot encoding
                # and concatenates this to the input features
                one_hot_length = cfg['output_dim']
                final_batch_x = []
                for features, reviews in zip(batch_x, padded_reviews):
                    for char_index in reviews[:-1]:
                        one_hot_encoding = np.zeros(one_hot_length)
                        one_hot_encoding[char_index] = 1
                        final_features = np.hstack((features, one_hot_encoding))
                        final_batch_x.append(final_features)

                # converts the final array into a numpy array
                final_batch_x = np.array(final_batch_x)

                # resizes the flattened array into batch_size x sequence_length x feature_length
                final_batch_x.resize(padded_reviews.shape[0], padded_reviews.shape[1] - 1, final_batch_x.shape[1])

                # converts final input array to tensors
                final_batch_x = torch.from_numpy(final_batch_x).float().to(computing_device)

                # passes the final input array to the model's forward pass
                outputs, _ = model(final_batch_x)
                soft_outputs = softmax(outputs)
                
                # prints the actual reviews vs the predicted reviews
                actual_reviews = batch_y
                predicted_reviews = decode_one_hot_reviews(soft_outputs)
                
                for a, p in zip(actual_reviews, predicted_reviews):
                    bleus.append(sentence_bleu(a.split(), p.split(), weights = [1.0], smoothing_function = bleu_smoothing.method1))
                    
                for i in range(1):
                    print ("Actual Review: " + actual_reviews[i])
                    print ("Predicted Review: " + predicted_reviews[i])


                # resizes the outputs to N x feature_length (for the loss function)
                outputs = outputs.view(-1, outputs.shape[2])

                # creates the targets and reshapes it to a single dimension
                targets = torch.from_numpy(padded_reviews[:, 1:]).long()
                targets = targets.view(-1)

                # passes the outputs and targets to the loss function
                loss = criterion(outputs, targets)

                losses.append(loss.item())

                print("Batch start index: " + str(start_index))
                print("Loss: " + str(loss.item()) + " | BLEU score: " + str(np.mean(bleus)))
                print("Time elapsed: " + str(time.time() - start_time))

                start_index = end_index
                end_index += cfg['batch_size']
                
                # 
                if end_index > len(x_valid):
                    break


        average_loss = np.mean(losses)
        valid_loss.append(average_loss)
        valid_bleu.append(np.mean(bleus))
        
        print()

        # checks for early stopping when the validation loss is higher for x consecutive epochs
        if average_loss >= min_loss:
            early_stop_count += 1

            if early_stop_count >= cfg['early_stop']:
                break

        else:
            early_stop_count = 0
            min_loss = average_loss
            
    return train_loss, valid_loss, valid_bleu
    
    
def process_results(model_name, train_loss, valid_loss, valid_bleu):
    # summarizes the results
    print (model_name + " Results:")
    print ("Training Loss: " + str(train_loss))
    print ("Validation Loss: " + str(valid_loss))
    print ("Validation Bleu Score: " + str(valid_bleu))

    # graphs the loss curves
    plt.clf()
    plt.plot(range(len(train_loss)), train_loss, 'b--', label = 'Training Loss')
    plt.plot(range(len(valid_loss)), valid_loss, 'r--', label = 'Validation Loss')

    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(model_name + " Loss Curve")
    plt.legend(loc="upper right")

    plt.savefig(model_name + " Loss.png")
    
    
    # graphs the bleu score curve
    plt.clf()
    plt.plot(range(len(valid_bleu)), valid_bleu, 'r--', label = 'Validation Bleu Score')
    
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Bleu Score")
    plt.title(model_name + " Bleu Score Curve")
    plt.legend(loc="lower right")

    plt.savefig(model_name + " Bleu Score.png")
    
    
def sample(outputs, temperature):
    logged = np.log(outputs) / temperature
    exped = np.exp(logged)
    sigmoided = exped / np.sum(exped)
    
    return np.random.multinomial(1, sigmoided)
    
def generate(model, x_test, cfg):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    
    predicted_reviews = []
    extended_char = cfg['valid_char'] + 'SEP'
    
    start_index = 0
    end_index = cfg['batch_size']
    
    start_time = time.time()

    print ('----- Testing -----')
    while start_index < len(x_test):
        # takes the minibatch subset
        batch_x = x_test[start_index:end_index]
        
        # sets the outputs as the <SOS> tag for each review
        outputs = np.zeros((cfg['batch_size'], cfg['output_dim']))
        outputs[:, cfg['valid_char_len']] = 1
        
        # initializes the states
        ht = None
        ct = None
        
        # initializes the predicted sentences
        sentences = [[] for _ in range(cfg['batch_size'])]
        
        # samples the next character until all are either <EOS> or <PAD> (all 1s are in the last 2 columns)
        while np.sum(outputs[:, :-2]) < cfg['batch_size'] and len(sentences[0]) < cfg['max_len']:
            # concatenates the outputs (previous characters) to the metadata to get the inputs
            final_batch_x = np.hstack((batch_x, outputs))
            
            # resizes the array into batch_size x sequence_length (1) x feature_length
            final_batch_x.resize(final_batch_x.shape[0], 1, final_batch_x.shape[1])

            # converts final input array to tensor
            final_batch_x = torch.from_numpy(final_batch_x).float().to(computing_device)

            # passes the final input array to the model's forward pass
            if isinstance(model, bLSTM):
                outputs, (ht, ct) = model(final_batch_x, ht, ct)
                
            else:
                outputs, ht = model(final_batch_x, ht)
                
            outputs = np.array([sample(c, cfg['gen_temp']) for c in outputs])
            
            sentences = np.hstack((sentences, [[np.argmax(c)] for c in outputs]))
            
            
        
        decoded = [''.join([extended_char[c] for c in review]) for review in sentences]
        predicted_reviews.append(decoded)
        '''
        for s in sentences:
            decoded = ''
            for c in s:
                if c < cfg['valid_char_len']:
                    decoded = decoded + cfg['valid_char'][c]
                    
                else:
                    break
                    
            predicted_reviews.append(decoded)
        '''

        #print ("Predicted Review: " + predicted_reviews[start_index])
        print ("Predicted Review: " + decoded[0])
            
        print("Batch start index: " + str(start_index))
        print("Time elapsed: " + str(time.time() - start_time))

        start_index = end_index
        end_index += cfg['batch_size']
        
        if start_index == len(x_test):
            break

        # case when the remaining data count is less than a minibatch
        if end_index > len(x_test):
            # adjusts the start and end indices to make the last subset the size of a minibatch
            end_index = len(x_test)
            start_index = end_index - cfg['batch_size']
            
            # removes the last few predictions to avoid duplicates
            predicted_reviews = predicted_reviews[:start_index]

    print()
    return predicted_reviews
    
    
def save_to_file(outputs, fname):
    # TODO: Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.
    raise NotImplementedError


### MAIN FUNCTION ###
if __name__ == "__main__":
    train_data_fname = "Beeradvocate_Train.csv"
    test_data_fname = "Beeradvocate_Test.csv"
    out_fname = "Output_Reviews.txt"

    # loads the data
    train_data = load_data(train_data_fname, cfg['num_data'])
    test_data = load_data(test_data_fname, 50)

    # processes the data to get the train, valid, and test sets
    train_data, train_labels, beer_to_index = process_train_data(train_data)
    x_train, y_train, x_valid, y_valid = train_valid_split(train_data, train_labels, cfg['train_percentage'])
    x_test = process_test_data(test_data, beer_to_index)

    # updates the configurations based on the processed data
    update_configurations(x_train.shape[1])

    # creates the dictionaries to map a character to its index in a one-hot encoding
    char_to_index = dict(zip(list(cfg['valid_char']), range(cfg['valid_char_len'])))

    # gets the computing device (either cuda or cpu)
    if torch.cuda.is_available():
        computing_device = torch.device("cuda")
        cfg['cuda'] = True

    else:
        computing_device = torch.device("cpu")
        cfg['cuda'] = False

    # defines the hyperparameters
    model_number = '1'

    cfg['hidden_dim'] = 16
    cfg['layers'] = 2

    # trains the LSTM model
    model = bLSTM(cfg).to(computing_device)
    optimizer = optim.Adam(model.parameters(), cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    train_loss, valid_loss, valid_bleu = train(model, "LSTM" + model_number, criterion, optimizer, computing_device, 
                                               x_train, y_train, x_valid, y_valid, cfg)

    process_results("LSTM Model " + model_number, train_loss, valid_loss, valid_bleu)

    predicted_reviews = generate(model, x_test, cfg)
    print (predicted_reviews)

    # trains the GRU model
    model = bGRU(cfg).to(computing_device)
    optimizer = optim.Adam(model.parameters(), cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    train_loss, valid_loss, valid_bleu = train(model, "GRU" + model_number, criterion, optimizer, computing_device, 
                                               x_train, y_train, x_valid, y_valid, cfg)

    process_results("GRU Model " + model_number, train_loss, valid_loss, valid_bleu)

    predicted_reviews = generate(model, x_test, cfg)
    print (predicted_reviews)

