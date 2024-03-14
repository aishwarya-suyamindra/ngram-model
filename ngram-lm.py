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

    # Add UNK to the vocab if there are tokens that occur only once and replace train tokens
    if 1 in tokens_dict.values():
       self.vocab.add(UNK)
    unk_tokens = [x for x in tokens_dict.keys() if tokens_dict[x] == 1]
    tokens = [t if t not in unk_tokens else UNK for t in tokens]

    # Count ngrams, n - 1 grams 
    elements = self.create_ngrams(tokens, self.n_gram)
    for elem in elements:
      key = tuple(elem)
      self.n_grams[key] = self.n_grams.get(key, 0) + 1    
    if self.n_gram > 1:
      elements = self.create_ngrams(tokens, self.n_gram - 1)
      for elem in elements:
        key = tuple(elem)
        self.n_minus_one_grams[key] = self.n_minus_one_grams.get(key, 0) + 1  

  def create_ngrams(self, tokens: list, n: int) -> list:
    """Computes and returns the n grams from the given list of tokens, using the vocab.
    Args:
      n (int): the n-gram order of tokens 
      tokens (list): tokenized data to be trained on
    Returns:
      list: the n-grams of order n
    """    
    ngrams = []
    for i in range(len(tokens) - n + 1):
      ngram_tokens = [t for t in tokens[i : i + n]]
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
    # replace tokens with UNK if it's not in the vocab
    unk_tokens = [x for x in sentence_tokens if x not in self.vocab]
    tokens = [t if t not in unk_tokens else UNK for t in sentence_tokens]

    # get ngrams and score 
    ngrams = self.create_ngrams(tokens, self.n_gram)
    for elem in ngrams:
      key = tuple(elem)
      probability_score *= self.__laplace_smoothing(key)
    return probability_score

  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    # set up start token context
    context = tuple(max(self.n_gram - 1, 1) * [SENTENCE_BEGIN])
    next_word = ""
    sentence = []
    i = self.n_gram
    sentence.extend(context)

    # generate until sentence end token is received
    while next_word != SENTENCE_END:
      next_word = self.__get_next_token(tuple(context))
      sentence.append(next_word)
      context = sentence[max(0, (i - self.n_gram) + 1): i]
      i += 1      
    return [" ".join(sentence)]
  
  def __get_next_token(self, history: tuple) -> str:
    """Returns the next word based on the given history. The next word is chosen randomly based on the ngram probability.
    Args:
      history (tuple): the n-1 context tokens
      
    Returns:
      str: the next token for the given history
    """ 
    if self.n_gram == 1:
      total_counts = sum(self.n_grams.values())
      # normalise the counts by the counts of the total ngrams for the context
      normalised_counts = [ [ngram[-1], (count / total_counts)] for (ngram, count) in self.n_grams.items() ]
    else:
      # get the possible ngrams for the given context 
      filtered_counts = { key: value for key, value in self.n_grams.items() if key[:-1] == history }
      if not filtered_counts:
        return SENTENCE_END
      # normalise the counts by the counts of the total ngrams for the context
      total_counts = sum(filtered_counts.values())
      normalised_counts = [ [ngram[-1], (count / total_counts)] for (ngram, count) in filtered_counts.items() ]
    next_token = np.random.choice([ngram_count[0] for ngram_count in normalised_counts], p = [ngram_count[1] for ngram_count in normalised_counts])
    return next_token

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
    sentences = language_model.generate(10)
    for sentence in sentences:
        sentence = sentence[0].replace("<s> ", "")
        sentence = sentence.replace(" </s>", "")
        print(sentence)
