import re
import math
import warnings
from collections import Counter
from scipy.sparse import csr_matrix

class BM25PlusVectorizer(object):
  """
  Implementation of Best Match-25 Plus algorithm

  Parameter(s):

    k1 : for tuning the impact of term-frequency, usually in range(1.2 to 2.0)

    b : for tuning the impact of doc length being scored, relative to average doc length 
        of the corpus, usually in range(0 to 1)

    delta : to lower bound the normalized term-frequency component, default = 1.0 
  
    stop_words : list of words(tokens) to be excluded at the time of pre-processing
    
    ngram_range : (a,b) create ngrams from n in range(a,b+1) as features
    
    token_pattern : tokenizes the documents(strings) based on this pattern

    tokenizer : user can pass a custom tokenizer function, then token_pattern has to be None 
    
    max_df : ignore terms which occur in #documents, greater than max_df threshold,
              if max_df is int, it is the absolute value of max_df threshold
              else, if max_df is float, max_df threshold = (max_df * #total documents)
    
    min_df : ignore terms which occur in #documents, lesser than min_df threshold,
              if min_df is int, it is the absolute value of min_df threshold
              else, if min_df is float, min_df threshold = (min_df * #total documents)
    
    max_features : consider only top max features ordered by final score across corpus
              
  """
  
  def __init__(self, k1 = 1.35, b = 0.75, delta = 1.0, 
                stop_words = None, ngram_range = (1,1), token_pattern = r'(?u)\b\w\w+\b', 
                tokenizer = None, max_df = 1.0, min_df = 1, max_features = None):

    self.k1 = k1
    self.b = b
    self.delta = delta
    self.stop_words = stop_words 
    self.ngram_range = ngram_range
    self.max_df = max_df
    self.min_df = min_df
    self.max_features = max_features
    self.tokenizer = tokenizer
    if tokenizer is not None:
      if token_pattern is not None:
        warnings.warn("tokenizer over-rides token_pattern. Set token_pattern=None explicitly to avoid warning.")
      self.token_pattern = None
    else:
      self.token_pattern = token_pattern

  
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
    self.ngrams_count = Counter()  # for term frequency
    self.doc_count = Counter()     # for document frequency
    clean_corpus = list()
    self.mean_doc_len = 0
    
    for document in corpus:
      # remove recurrent spaces
      document = re.sub(r'\s+', ' ', document.lower())
      # tokenize on the basis of given pattern or custom tokenizer
      tokens = re.findall(self.token_pattern, document) if self.tokenizer is None else self.tokenizer(document)
      if self.stop_words is not None and len(self.stop_words) > 0:
        # filter stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

      # create ngrams for later featurization and doc frequency
      for n in range(self.ngram_range[0], self.ngram_range[1]+1):
        ngrams = self.create_ngrams(tokens, n)
        self.ngrams_count.update(ngrams)
        self.doc_count.update(set(ngrams))
      
      # create vocabulary
      self.vocabulary.update(tokens)
      clean_corpus.append(' '.join(tokens))
      self.mean_doc_len += len(tokens)
      
    self.mean_doc_len /= len(corpus)
    return clean_corpus

  def fit(self, corpus):
    """
    Fits the BM25Plus vectorizer over given corpus
    
    Parameter(s):
      corpus : list of strings, where each string is considered a document
    """
    
    self.clean_corpus = self.preprocess(corpus)
    self.idf_score = dict()
    # calculate min_df and max_df thresholds from their given values
    min_df_thres = self.min_df if type(self.min_df) is int else int(self.min_df*len(corpus))
    max_df_thres = self.max_df if type(self.max_df) is int else int(self.max_df*len(corpus))
    
    for phrase in self.ngrams_count.keys():
      doc_count = self.doc_count.get(phrase, 0)
      if doc_count >= min_df_thres and doc_count <= max_df_thres:
        # calculate idf score 
        self.idf_score[phrase] = math.log(((len(corpus)-doc_count+0.5)/(doc_count+0.5)) +1)
    
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
    
    assert hasattr(self, 'features'), "BM25+ Vectorizer has to be fitted first, run fit() method."
    
    row_i, col_i, bm25p_score = list(), list(), list()
    for di,document in enumerate(self.clean_corpus):
      for fi,feat in enumerate(self.features):
        feat_count = len(re.findall('\\b%s\\b'%feat, document))
        if feat_count > 0:
          # calculate the term frequency ratio score
          doc_len, feat_len = len(document.split()), len(feat.split())
          total_count = doc_len - feat_len + 1.0
          tf_ratio = feat_count / total_count
          numerator = tf_ratio * (self.k1 + 1.0)
          denominator = tf_ratio + self.k1 * (1.0 - self.b + (self.b * doc_len/self.mean_doc_len))
          tf_score = (numerator/denominator) + self.delta
          row_i.append(di)
          col_i.append(fi)
          bm25p_score.append(tf_score * self.idf_score[feat])
    
    bm25p_matrix = csr_matrix((bm25p_score, (row_i, col_i)), shape=(len(self.clean_corpus),
                                                            len(self.features)), dtype=float)
    return bm25p_matrix

if __name__ == '__main__':
  corpus = [
        'this is the first document',
        'this document is the second document',
        'and this is the third one',
        'is this the first document'
      ] 

  bm25p = BM25PlusVectorizer(stop_words=['this','is', 'the'], ngram_range=(1,2), max_features=3)
  bm25p.fit(corpus)
  print(bm25p.transform().toarray())
  print(bm25p.features)
  print(bm25p.ngrams_count)
