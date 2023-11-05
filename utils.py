import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw')
nltk.download('words')
random.seed(0)
nltk.download('stopwords')
nltk.download('punkt')
def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.
def generate_typo(sentence):
    qwerty_keyboard = {
        'a': 'qwsz',
        'b': 'vghn',
        'c': 'xdfv',
        'd': 'erfvcx',
        'e': 'rwsd',
        'f': 'rtgvcd',
        'g': 'tyhbvf',
        'h': 'yujngb',
        'i': 'uoik',
        'j': 'yuihkn',
        'k': 'uiolmj',
        'l': 'opk',
        'm': 'njk',
        'n': 'bhjm',
        'o': 'iklp',
        'p': 'ol',
        'q': 'aws',
        'r': 'edft',
        's': 'qawedxz',
        't': 'rfgy',
        'u': 'yhjik',
        'v': 'cfgb',
        'w': 'qesa',
        'x': 'zcd',
        'y': 'tghu',
        'z': 'xsa',
    }

    typo_sentence = list(sentence)
    index = random.randint(0, len(typo_sentence) - 1)
    letter = typo_sentence[index]
    if letter in qwerty_keyboard:
        typo_options = qwerty_keyboard[letter]
        if typo_options:
            typo_letter = random.choice(typo_options)
            typo_sentence[index] = typo_letter

    return ''.join(typo_sentence)

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.
    stop_words = nltk.corpus.stopwords.words('english')

    # You should update example["text"] using your transformation
    rand = random.randint(0, 1)
    if(rand):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        sentences = nltk.sent_tokenize(example['text'])
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            stopword_indices = [j for j, word in enumerate(words) if word in stop_words]
            max_removals = min(7, len(stopword_indices))
            for _ in range(max_removals):
                if stopword_indices:
                    random_index = random.choice(stopword_indices)
                    if random_index < len(words):
                        words.pop(random_index)
                        stopword_indices.remove(random_index)
            sentences[i] = ' '.join(words)
        example['text'] = ' '.join(sentences)
        return example
    else:
        '''
        qwerty_keyboard = {
            'a': 'qwsz',
            'b': 'vghn',
            'c': 'xdfv',
            'd': 'erfvcx',
            'e': 'rwsd',
            'f': 'rtgvcd',
            'g': 'tyhbvf',
            'h': 'yujngb',
            'i': 'uoik',
            'j': 'yuihkn',
            'k': 'uiolmj',
            'l': 'opk',
            'm': 'njk',
            'n': 'bhjm',
            'o': 'iklp',
            'p': 'ol',
            'q': 'aws',
            'r': 'edft',
            's': 'qawedxz',
            't': 'rfgy',
            'u': 'yhjik',
            'v': 'cfgb',
            'w': 'qesa',
            'x': 'zcd',
            'y': 'tghu',
            'z': 'xsa',
        }

        typo_example = list(example['text'])
        index = random.randint(0, len(typo_example) - 1)
        letter = typo_example[index]
        if letter in qwerty_keyboard:
            typo_options = qwerty_keyboard[letter]
            if typo_options:
                typo_letter = random.choice(typo_options)
                typo_example[index] = typo_letter
        example['text'] = ''.join(typo_example)
        return example
    '''
        stop_words = nltk.corpus.stopwords.words('english')
        sentences = nltk.sent_tokenize(example['text'])
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            num_replacements = random.randint(1, 9)
            for _ in range(num_replacements):
                word_indices = list(range(len(words)))
                random.shuffle(word_indices)
                for index in word_indices:
                    word = words[index]
                    if word.lower() not in stop_words:
                        new_word = generate_typo(word)  
                        words[index] = new_word
                        break 
            sentences[i] = ' '.join(words)
        example['text'] = ' '.join(sentences)
    return example

    ##### YOUR CODE ENDS HERE ######

    return example
