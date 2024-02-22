from collections import Counter
import numpy as np
import random
import csv

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"

class LanguageModel:

  def __init__(self, n_gram):
    """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    self.n_gram = n_gram
    self.vocab = set()
    self.n_grams = {}
    self.n_minus_one_grams = {}
    self.ngram_probability = {}
    self.train_tokens = 0
  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    # Build the tokens dictionary and vocabulary
    self.train_tokens = len(tokens)
    tokens_dict = dict(Counter(tokens))
    self.vocab = set(key for key in tokens_dict.keys() if tokens_dict[key] != 1)
    # Add UNK to the vocab if there are tokens that occur only once
    if 1 in tokens_dict.values():
       self.vocab.add(UNK)

    # Count ngrams, n - 1 grams 
    unk_tokens = [x for x in tokens_dict.keys() if tokens_dict[x] == 1]
    elements = self.__get_n_grams__(self.n_gram, tokens, unk_tokens)
    for elem in elements:
      key = tuple(elem)
      self.n_grams[key] = self.n_grams.get(key, 0) + 1    
    if self.n_gram > 1:
      elements = self.__get_n_grams__(self.n_gram - 1, tokens, unk_tokens)
      for elem in elements:
        key = tuple(elem)
        self.n_minus_one_grams[key] = self.n_minus_one_grams.get(key, 0) + 1      
    
    # Calculate the probabilities of n-grams
    for token in self.n_grams.keys():
      self.ngram_probability[token] = self.__laplace_smoothing__(len(tokens), token)

  def __get_n_grams__(self, n: int, tokens: list, unk_tokens: list) -> list:
    """Computes and returns the n grams from the given list of tokens, using the vocab.
    Args:
      n (int): the n-gram order of tokens 
      tokens (list): tokenized data to be trained on
      unk_tokens (list): the unknown tokens
    Returns:
      list: the n-grams of order n
    """    
    ngrams = []
    for i in range(len(tokens) - n + 1):
      ngram_tokens = [t if t not in unk_tokens else UNK for t in tokens[i : i + n]]
      ngrams.append(ngram_tokens)
    return ngrams

    
  def __laplace_smoothing__(self, n: int, token: tuple) -> float:
    """Adds one to all the n-gram counts and normalises to probabilities.
    Args:
      n (int): the number of tokens
      token (tuple): the n-gram
    Returns:
      float: the add one smoothed probability of the n-gram
    """
    if self.n_gram == 1:
      return (self.n_grams.get(token, 0) + 1) / (self.train_tokens + len(self.vocab))
    else:
      n_minus_one_gram = tuple(token[0:len(token) - 1])
      return (self.n_grams.get(token, 0) + 1) / (self.n_minus_one_grams.get(n_minus_one_gram, 0) + len(self.vocab))
  

  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    probability_score = 1
    unk_tokens = [x for x in sentence_tokens if x not in self.vocab]
    ngrams = self.__get_n_grams__(self.n_gram, sentence_tokens, unk_tokens)
    for elem in ngrams:
      key = tuple(elem)
      if key in self.ngram_probability:
        probability_score *= self.ngram_probability[key]
      else:
        probability_score *= self.__laplace_smoothing__(len(sentence_tokens), key)
    return probability_score

  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    context = tuple(max(self.n_gram - 1, 1) * [SENTENCE_BEGIN])
    next_word = ""
    sentence = []
    i = self.n_gram
    sentence.extend(context)
    while next_word is not SENTENCE_END:
      next_word = self.__get_next_word__(tuple(context))
      sentence.append(next_word)
      context = sentence[max(0, (i - self.n_gram) + 1): i]
      i += 1      
    return [" ".join(token for token in sentence if token is not UNK)]
  
  def __get_next_word__(self, history: tuple) -> str:
    """Returns the next word based on the given history. The next word is chosen randomly based on the ngram probability.
    Args:
      history (tuple): the n-1 context tokens
      
    Returns:
      str: the next word for the given history
    """
    probabilities = {}
    if self.n_gram == 1:
      probabilities = {key: self.ngram_probability[key] for key in self.ngram_probability.keys() if key != ('<s>',) }
    else:
      probabilities = {key: self.ngram_probability[key] for key in self.ngram_probability.keys() if key[0 : self.n_gram - 1] == history}
    return random.choice(list(probabilities.keys()))[-1]

  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    return [self.generate_sentence() for i in range(n)]


class Tokenizer:
    def __init__(self, by_character: bool = False):
        self.by_character = by_character

    def tokenize_line(self, line: str, ngram: int, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
        """
        Tokenize a single string. Glue on the appropriate number of 
        sentence begin tokens and sentence end tokens (ngram - 1), except
        for the case when ngram == 1, when there will be one sentence begin
        and one sentence end token.
        Args:
          line (str): text to tokenize
          ngram (int): ngram
          sentence_begin (str): sentence begin token value
          sentence_end (str): sentence end token value

        Returns:
          list of strings - a single line tokenized
        """
        inner_pieces = None
        if self.by_character:
          inner_pieces = list(line)
        else:
          # split on white space
          inner_pieces = line.split()

        if ngram == 1:
          tokens = [sentence_begin] + inner_pieces + [sentence_end]
        else:
          tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
        return tokens
    
    def tokenize(self, data: list, ngram: int, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
        """
        Tokenize each line in a list of strings. Glue on the appropriate number of 
        sentence begin tokens and sentence end tokens (ngram - 1), except
        for the case when ngram == 1, when there will be one sentence begin
        and one sentence end token.
        Args:
            data (list): list of strings to tokenize
            ngram (int): ngram
            sentence_begin (str): sentence begin token value
            sentence_end (str): sentence end token value

        Returns:
        list of strings - lines tokenized as one large list
        """
        tokens = []
        for line in data:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens += self.tokenize_line(line, ngram, sentence_begin, sentence_end)
        return tokens
    
# if __name__ == '__main__':
#     # load and process data
#     file_path = "data/database_main_socrates.json"
#     train_data = []
#     with open(file_path, "r") as file:
#         data = json.loads(file.read())
#     for entry in data:
#         output = entry["output"]
#         for line in output.split("."):
#           line = line.strip().replace("\n", "").replace("- ", "").replace('"', "")
#           train_data.append(line)
    
#     # Tokenize
#     ngram = 3
#     tokenizer = Tokenizer()
#     tokens = tokenizer.tokenize(train_data, ngram)

#     # Train the ngram model
#     language_model = LanguageModel(ngram)
#     language_model.train(tokens)

#     print

#     # Generate sentences
#     sentences = language_model.generate(10)
#     for sentence in sentences:
#         sentence = sentence[0].replace("<s> ", "")
#         sentence = sentence.replace(" </s>", "")
#         print(sentence)

if __name__ == '__main__':
    # load and process data
    file_path = "data/sequences.csv"
    train_data = []
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data = row["Input Dialog"] + row["Output Dialog"]
            for line in data.split("."):
                line = line.strip().replace("\n", "").replace("- ", "").replace('"', "")
                train_data.append(line)
    
    # Tokenize
    ngram = 5
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(train_data, ngram)

    # Train the ngram model
    language_model = LanguageModel(ngram)
    language_model.train(tokens)

    # Generate sentences
    sentences = language_model.generate(20)
    for sentence in sentences:
        sentence = sentence[0].replace("<s> ", "")
        sentence = sentence.replace(" </s>", "")
        print(sentence)




    

