import pandas as pd
from nltk import tokenize
import nltk
from string import punctuation
from nltk.corpus import stopwords
# nltk.download('punkt')
from tqdm import tqdm_notebook
import re

nltk.download('stopwords')
df = pd.read_excel(r"C:\Users\shaik\Downloads\Input (4).xlsx")
positive_words = set(open(r"C:\Users\shaik\Downloads\positive-words.txt").read().splitlines())
negative_words = set(open(r"C:\Users\shaik\Downloads\negative-words.txt").read().splitlines())
stopwords = set(open(r"C:\Users\shaik\Desktop\merged_stopwords3.txt").read().splitlines())

df['positive_score'] = 0
df['negative_score'] = 0
df['words_after_cleaning'] = 0
for i, row in df.iterrows():
    words = row['description'].lower().split()
    positive_score = sum([1 for word in words if (word in positive_words) and not(word in stopwords)])
    negative_score = sum([1 for word in words if (word in negative_words) and not(word in stopwords)])
    words_after_cleaning = sum([1 for word in words if  not(word in stopwords)])

    df.at[i, 'positive_score'] = positive_score
    df.at[i, 'negative_score'] = negative_score
    df.at[i, 'words_after_cleaning'] = words_after_cleaning
    df.at[i, 'clean_text'] = row['description'].translate(str.maketrans("", "", punctuation))
df['polarity_score'] = (df.positive_score - df.negative_score)/(df.positive_score + df.negative_score + 0.000001)
df['subjectivity_score'] = (df.positive_score + df.negative_score)/(df.words_after_cleaning + 0.000001)
df['avg_no_of_words_per_sentence'] = df.description.apply(lambda p: len(p.split())/len(tokenize.sent_tokenize(p)))
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

stop_words = stopwords.words('english')
df['word_count'] = df.clean_text.apply(lambda textt : sum([1 for word in nltk.word_tokenize(textt) if  not(word in stop_words)]))
df['syllable_count'] = 0
df['avg_word_length'] = 0
for index, row in tqdm_notebook(df.iterrows()):
    syl_count = 0
    word_cnt = 0
    letter_count = 0
    for word in row.description.lower().split():
        word_cnt+=1
        for letter in word.strip("es").strip("ed"):
            letter_count+=1
            if letter in ['a','e','i','o','u']:
                syl_count+=1
    df.at[index, 'syllable_count'] = syl_count
    df.at[index, 'avg_word_length'] = letter_count/word_cnt
df['personal_pronouns'] = df.description.apply(lambda x : len(re.findall("I|we|us|ours|my",x)))
df['avg_sentence_length'] = df['description'].apply(lambda x: len(nltk.word_tokenize(x))/len(nltk.sent_tokenize(x)))
def count_syllables(word):
    vowels = "aeiouy"
    word = word.lower().strip(".:;?!")

    if len(word) == 0:
        return 0

    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
# Apply the function to each word in the description column to count the number of complex words
df['complex_word_count'] = df['description'].apply(lambda x: sum(1 for word in nltk.word_tokenize(x) if count_syllables(word) > 2))

df['percentage_complex_words'] = df['complex_word_count'] / df['word_count'] * 100
df['fog_index'] = 0.4 * (df['avg_sentence_length'] + df['percentage_complex_words'])
df = df.drop(['WORDS_AFTER_CLEANING', 'CLEAN_TEXT', 'DESCRIPTION'], axis=1)

df.columns = map(str.upper, df.columns)
df.to_excel("word.xlsx", index = False)

