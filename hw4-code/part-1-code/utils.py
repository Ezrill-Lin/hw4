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
import nltk

# Download required NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords

random.seed(0)


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

stop_words = set(stopwords.words("english"))

def synonym_replacement(sentence, prob=0.5):
    words = word_tokenize(sentence)
    new_words = []
    for word in words:
        if word.lower() in stop_words or not word.isalpha():
            new_words.append(word)
            continue
        # Randomly decide to replace with much higher probability
        if random.random() < prob:
            synsets = wordnet.synsets(word)
            if synsets:
                # Try to get synonyms from multiple synsets for more variation
                all_synonyms = []
                for synset in synsets[:3]:  # Use first 3 synsets
                    lemmas = synset.lemmas()
                    all_synonyms.extend([lemma.name().replace('_', ' ') 
                                for lemma in lemmas if lemma.name().lower() != word.lower()])
                if all_synonyms:
                    new_word = random.choice(all_synonyms)
                    new_words.append(new_word)
                    continue
        new_words.append(word)
    return ' '.join(new_words)

keyboard_neighbors = {
    'a': ['s','q','w','z'],
    'b': ['v','g','h','n'],
    'c': ['x','d','f','v'],
    'd': ['s','e','r','f','c','x'],
    'e': ['w','s','d','r'],
    'f': ['d','r','t','g','v','c'],
    'g': ['f','t','y','h','b','v'],
    'h': ['g','y','u','j','n','b'],
    'i': ['u','j','k','o'],
    'j': ['h','u','i','k','n','m'],
    'k': ['j','i','o','l','m'],
    'l': ['k','o','p'],
    'm': ['n','j','k'],
    'n': ['b','h','j','m'],
    'o': ['i','k','l','p'],
    'p': ['o','l'],
    'q': ['w','a'],
    'r': ['e','d','f','t'],
    's': ['a','w','e','d','x','z'],
    't': ['r','f','g','y'],
    'u': ['y','h','j','i'],
    'v': ['c','f','g','b'],
    'w': ['q','a','s','e'],
    'x': ['z','s','d','c'],
    'y': ['t','g','h','u'],
    'z': ['a','s','x'],
}

def typo_transform(sentence, prob=0.25, typo_per_word=0.4):
    words = sentence.split()
    new_words = []
    for word in words:
        if random.random() < prob and len(word) > 2:
            chars = list(word)
            # Introduce multiple typos per word for stronger effect
            num_typos = max(1, int(len(chars) * typo_per_word))
            indices = random.sample(range(len(chars)), min(num_typos, len(chars)))
            
            for idx in indices:
                c = chars[idx].lower()
                if c in keyboard_neighbors:
                    chars[idx] = random.choice(keyboard_neighbors[c])
            word = ''.join(chars)
        new_words.append(word)
    return ' '.join(new_words)

def custom_transform(example, typo=True, prob=0.5):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    if typo:
        example["text"] = typo_transform(example["text"], prob=0.25)
    else:
        example["text"] = synonym_replacement(example["text"], prob)

    ##### YOUR CODE ENDS HERE ######

    return example
