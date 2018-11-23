'''
# -*- coding: utf-8 -*-
All kind of text cleaning operations, can be used for
machine learning algorithms

'''
# remove_punctuation, input is a word
def remove_punctuation_word(word):
    import string
    exclude = set(string.punctuation)
    return ''.join(ch for ch in word if ch not in exclude)

#!/usr/bin/python3,input is a sentence
def remove_punctuation(text):
    import string
    return text.translate(str.maketrans('','',string.punctuation))

# remove numbers associated with text, ex: "total5"
def remove_numbers(text):
    return text.translate(str.maketrans('','','0123456789'))


# one-shot text cleaning approach
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    import re
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# Approach to punctuation would preserve phrases such as "I'd", "would've", "hadn't"
'''
'''
# Text pre-processing
def text_preprocessing(df,col_in,col_out):
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    from textblob import TextBlob
    from nltk.stem import PorterStemmer
    st = PorterStemmer()
    from textblob import Word
    from tqdm import tqdm
    
    #df is dataframe
    # col_in: input column used for processing
    # col_out: output column name
    # return df
    
    tqdm.pandas(desc="lower_case")
    # convert to lower_case
    df[col_out] = df[col_in].progress_apply(lambda x : str(x).lower())
    print("\n")
    tqdm.pandas(desc="Punctuation")
    # Removing Punctuation
    df[col_out] = df[col_out].progress_apply(lambda x : remove_punctuation(str(x)))
    print("\n")
    tqdm.pandas(desc="Stop_Words")
    # Removal of Stop Words
    df[col_out] = df[col_out].progress_apply(lambda x : " ".join([i for i in x.split() if i not in stop]))
    print("\n")
    tqdm.pandas(desc="Spelling_correction")
    # Spelling correction
    df[col_out] = df[col_out].progress_apply(lambda x : str(TextBlob(x).correct()))
    print("\n")
    tqdm.pandas(desc="Stemming")
    # Stemming
    df[col_out] = df[col_out].progress_apply(lambda x : " ".join([st.stem(word) for word in x.split()]))
    print("\n")
    tqdm.pandas(desc="Lemmatization")
    # Lemmatization
    df[col_out] = df[col_out].progress_apply(lambda x: " ".join([Word(word).lemmatize("v") for word in x.split()]))
    
    return df
    
