import re
import math
from collections import Counter
from scipy.sparse import csr_matrix


class TfidfVectorizer(object):
  """
  Simple implementation of Tf-Idf Vectorizer

  Parameter(s):
  
    stop_words : list of words(tokens) to be excluded at the time of pre-processing
    
    ngram_range : (a,b) create ngrams from n in range(a,b+1) as features
    
    token_pattern : tokenizes the documents(strings) based on this pattern
    
    max_df : ignore terms which occur in #documents, greater than max_df threshold,
              if max_df is int, it is the absolute value of max_df threshold
              else, if max_df is float, max_df threshold = (max_df * #total documents)
    
    min_df : ignore terms which occur in #documents, lesser than min_df threshold,
              if min_df is int, it is the absolute value of min_df threshold
              else, if min_df is float, min_df threshold = (min_df * #total documents)
    
    max_features : consider only top max features ordered by frequency across corpus
              
  """
  
  def __init__(self, stop_words = None, ngram_range = (1,1), token_pattern = r'(?u)\b\w\w+\b',
                     max_df = 1.0, min_df = 1, max_features = None):
    
    self.stop_words = stop_words 
    self.ngram_range = ngram_range
    self.token_pattern = token_pattern
    self.max_df = max_df
    self.min_df = min_df
    self.max_features = max_features
  
  def create_ngrams(self, tokens, n):
    """
    Return ngrams of given tokens
    
    Parameter(s):
      tokens : list of strings, each string is a document word
      n : n in n-gram
    """
    
    assert len(tokens) >= n, "%d-grams not possible with %d tokens" % (n, len(tokens))
    return [' '.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)])]
  
  def preprocess(self, corpus):
    """
    Return a cleaned version of given corpus
    
    Parameter(s):
      corpus : list of strings, where each string is considered a document
    """
    
    self.vocabulary = set()
    self.ngrams_count = Counter()
    clean_corpus = list()
    for document in corpus:
      # remove recurrent spaces
      document = re.sub(r'\s+', ' ', document.lower())
      # tokenize on the basis of given pattern
      tokens = re.findall(self.token_pattern, document)
      if self.stop_words is not None and len(self.stop_words) > 0:
        # filter stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

      # create ngrams for later featurization
      for n in range(self.ngram_range[0], self.ngram_range[1]+1):
        self.ngrams_count.update(self.create_ngrams(tokens, n))
      # create vocabulary
      self.vocabulary.update(tokens)
      clean_corpus.append(' '.join(tokens))
    
    return clean_corpus

  def fit(self, corpus):
    """
    Fits the Tf-Idf vectorizer over given corpus
    
    Parameter(s):
      corpus : list of strings, where each string is considered a document
    """
    
    self.clean_corpus = self.preprocess(corpus)
    self.idf_score = dict()
    # calculate min_df and max_df thresholds from their given values
    min_df_thres = self.min_df if type(self.min_df) is int else int(self.min_df*len(corpus))
    max_df_thres = self.max_df if type(self.max_df) is int else int(self.max_df*len(corpus))
    for phrase in self.ngrams_count.keys():
      doc_count = sum([1 for document in self.clean_corpus if re.search('\\b%s\\b'%phrase, document)])
      if doc_count >= min_df_thres and doc_count <= max_df_thres:
        # calculate idf score 
        self.idf_score[phrase] = (math.log((1+len(corpus))/(doc_count+1)) +1)
    # get the features as per given limit
    self.features = [phrase for phrase,count in self.ngrams_count.most_common() if phrase in self.idf_score]
    if self.max_features is not None:
      self.features = self.features[:self.max_features]

  def transform(self):
    """
    Transforms the given corpus
    
    Parameter(s):
      None
    """
    
    assert hasattr(self, 'features'), "Tf-Idf Vectorizer has to be fitted first, run fit() method."
    
    row_i, col_i, tf_idf_score = list(), list(), list()
    for di,document in enumerate(self.clean_corpus):
      for fi,feat in enumerate(self.features):
        feat_count = len(re.findall('\\b%s\\b'%feat, document))
        if feat_count > 0:
          # calculate the term frequency ratio score
          total_count = len(self.create_ngrams(document.split(), len(feat.split())))
          tf_score = feat_count / total_count
          row_i.append(di)
          col_i.append(fi)
          tf_idf_score.append(tf_score * self.idf_score[feat])
    
    tf_idf_matrix = csr_matrix((tf_idf_score, (row_i, col_i)), 
                               shape=(len(self.clean_corpus),len(self.features)), dtype=float)
    return tf_idf_matrix


if __name__ == '__main__':
  corpus = [
        'this is the first document',
        'this document is the second document',
        'and this is the third one',
        'is this the first document'
      ] 

  tfidf = TfidfVectorizer(stop_words=['this','is', 'the'], ngram_range=(1,2), max_features=3)
  tfidf.fit(corpus)
  print(tfidf.transform().toarray())
  print(tfidf.features)
  print(tfidf.ngrams_count)
