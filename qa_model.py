import nltk
nltk.download('punkt')
from functools import reduce
import tarfile
import numpy as np
import string
import re

import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM, GRU
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


#Hyperparams
train_epochs = 100
batch_size = 32
lstm_size = 64

def tokenize(sent):
    tokens = nltk.word_tokenize(sent)
    return tokens


def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
           
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    try:
        for story, query, answer in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in query]
            y = np.zeros(len(word_idx) + 1)
            y[word_idx[answer]] = 1
            X.append(x)
            Xq.append(xq)
            Y.append(y)
    except:
        print("-----------------------------------------------\n")
        print("Couldn't answer this question\n")
        print("-----------------------------------------------\n")
        return False
    return (pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


"""
try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
    print(path)
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
"""

path = 'tasks_1-20_v1-2.tar.gz'
tar = tarfile.open(path)

challenges = {
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    'three_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_{}.txt',
    'agents_motivations_10k' : 'tasks_1-20_v1-2/en-10k/qa20_agents-motivations_{}.txt',
    'basic_deduction_10k' : 'tasks_1-20_v1-2/en-10k/qa15_basic-deduction_{}.txt',
}
challenge_type = ('single_supporting_fact_10k')
challenge = challenges[challenge_type]

#print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

"""
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')
"""

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
idx_word = dict((i+1, c) for i,c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_maxlen,
                                                               query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_maxlen,
                                                            query_maxlen)

"""
print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
"""

input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

#print('Input sequence:', input_sequence)
#print('Question:', question)

input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
input_encoder_m.add(Dropout(0.3))

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
input_encoder_c.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=query_maxlen))
question_encoder.add(Dropout(0.3))
input_encoded_m = input_encoder_m(input_sequence)
print('Input encoded m', input_encoded_m)
input_encoded_c = input_encoder_c(input_sequence)
print('Input encoded c', input_encoded_c)
question_encoded = question_encoder(question)
print('Question encoded', question_encoded)

match = dot([input_encoded_m, question_encoded], axes=(2, 2))
print(match.shape)
match = Activation('softmax')(match)
print('Match shape', match)

response = add([match, input_encoded_c]) 
response = Permute((2, 1))(response)  
print('Response shape', response)

answer = concatenate([response, question_encoded])
print('Answer shape', answer)
answer = LSTM(lstm_size)(answer)
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size)(answer) 

answer = Activation('softmax')(answer)
model = Model([input_sequence, question], answer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("-------------Model Summary------------")
print(model.summary())

print("Training the model")
model.fit([inputs_train, queries_train], answers_train, batch_size, train_epochs, validation_data=([inputs_test, queries_test], answers_test))
model.save('model.h5')


while 1:
    print('-------------------------------------------------------------------------------------------')
    user_story_inp = '.'
    print('Please input a query')
    user_query_inp = input().split(' ')
    user_story, user_query, user_ans = vectorize_stories([[user_story_inp, user_query_inp, '.']], word_idx, story_maxlen, query_maxlen)
    user_prediction = model.predict([user_story, user_query])
    user_prediction = idx_word[np.argmax(user_prediction)]
    print('Result:')
    print(' '.join(user_query_inp), '| Prediction:', user_prediction)
