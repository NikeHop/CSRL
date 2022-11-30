import string

from nltk import ngrams, word_tokenize

translator = str.maketrans('', '', string.punctuation)


def tokenize(text: str):
    return word_tokenize(text.lower().translate(translator).strip())


def is_substring(text: str, elements: set) -> bool:
    """
    Check if a string is a substring of any string in a set

    Args:
        text (str): text to be tested
        elements (set(str)): set of string to be tested against for substring condition

    Return:
        (bool): whether or not if text is a substring of strings in elements
    """
    for element in elements:
        if text in element:
            return True

    return False


def get_extractor(token_extractor_type: str):
    if token_extractor_type == 'any':
        return any_substring_extraction
    
    elif token_extractor_type == 'max_bag_of_words':
        return max_substring_extraction_bag
    
    elif token_extractor_type =='max':
        return max_substring_extraction
    
    return max_substring_extraction

def max_substring_extraction_bag(text: str, entities: set, ngram: int = 3, stopwords: set = None) -> set:
        """
        The function extract all valid entities based on maximum substring policy
        Args:
            text (str): string from which entities to be extracted
            entities (set): set of all valid entities
            n (int): size of n-gram
            stopwords (set): set of words to be ignored
        Returns:
            set(str): set containing extracted entitites
        """
        candidates = set()
        
        # Iterative build N-grams with reducing N and preserve biggest ngram entities
        
        for N in range(ngram, 0, -1):
            for tokens in ngrams(tokenize(text), N):
                entity = '_'.join(tokens)
                if entity in entities:
                    if (stopwords and entity in stopwords.words()) or is_substring(entity, candidates) :
                        continue
                    candidates.add(entity)
        
        return candidates

def max_substring_extraction(text: str, entities: set, ngram: int = 3, stopwords: set = None) -> list:
        """
        The function extract all valid entities based on maximum substring policy
        Args:
            text (str): string from which entities to be extracted
            entities (set): set of all valid entities
            n (int): size of n-gram
            stopwords (set): set of words to be ignored
        Returns:
            set(str): set containing extracted entitites
        """
        candidates = []
        text = tokenize(text)

        n_gram_list = [list(ngrams(text,i)) for i in range(1,ngram+1)]
        
        while len(n_gram_list[0])>0:
            # Start with the largest n-grams
            for i in reversed(range(ngram)):
                # If the list is empty move to the next smaller one
                if len(n_gram_list[i])==0:
                    continue
                trial_string = "_".join(n_gram_list[i][0])
                # If the string is in the vocab, update the n-gram lists and 
                # break out of for loop
                if trial_string in entities:
                    # Do not add it to the list of words, if it is a stopword or already present
                    if (stopwords and trial_string in stopwords.words()) or is_substring(trial_string, candidates):
                        if i==0:
                            n_gram_list = [l[1:] for l in n_gram_list]
                        continue
                    candidates.append(trial_string)
                    # smaller n-grams are always contained in the larger ones
                    n_gram_list = [l[1:] for l in n_gram_list]
                    break
                if i==0:
                    n_gram_list = [l[1:] for l in n_gram_list]
        return candidates


def any_substring_extraction(text: str, entities: set, ngram: int = 3, stopwords: set = None) -> set:
    candidates = set()
    for N in range(ngram, 0, -1):
        for tokens in ngrams(tokenize(text), N):
            entity = '_'.join(tokens)
            if entity in entities:
                if stopwords and entity in stopwords:
                    continue
                candidates.add(entity)

    return candidates
