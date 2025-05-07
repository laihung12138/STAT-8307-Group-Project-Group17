import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud, STOPWORDS
import nltk
from textblob import TextBlob
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade, difficult_words, dale_chall_readability_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import string

from models import split_raw_data_to_csv_dis_lora
def load_data(essays_path):
    """Load and split the dataset into train and test sets."""
    all_essays_df = pd.read_csv(essays_path)
    train_essays_df, test_essays_df = train_test_split(all_essays_df, test_size=0.2, random_state=42)
    return train_essays_df, test_essays_df

#def clean_data(df):
#    """Clean the dataset by checking for missing values and duplicates."""
#    # Check for missing values
#    missing_values = df.isnull().sum()
#    print("Missing values:\n", missing_values)
    
#    # Check for duplicates
#    if df['text'].duplicated().any():
#        df = df.drop_duplicates(subset=['text'])
#        print("Duplicates removed.")
#    else:
#        print("No duplicates found.")
#
#    return df

def analyze_text_length(df):
    """Analyze and visualize text length distribution."""
    df['essay_length'] = df['text'].apply(len)
    
    # Descriptive statistics
    human_stats = df[df['label'] == 0]['essay_length'].describe()
    llm_stats = df[df['label'] == 1]['essay_length'].describe()
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='label', y='essay_length', data=df)
    plt.grid(False)
    plt.xlabel('Text Source', fontsize=20)
    plt.ylabel('Text Length', fontsize=20)
    plt.xticks([0, 1], ['Student-written', 'LLM-generated'], fontsize=20)
    plt.show()
    
    return human_stats, llm_stats

def calculate_text_metrics(df):
    """Calculate various text metrics."""
    def calculate_metrics(text):
        words = text.split()
        sentences = text.split('.')
        word_count = len(words)
        unique_word_count = len(set(words))
        sentence_count = len(sentences)
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        return word_count, unique_word_count, sentence_count, avg_word_length
    
    df['metrics'] = df['text'].apply(calculate_metrics)
    df[['word_count', 'unique_word_count', 'sentence_count', 'avg_word_length']] = pd.DataFrame(df['metrics'].tolist(), index=df.index)
    df.drop('metrics', axis=1, inplace=True)
    
    comparison_metrics = df.groupby('label')[['word_count', 'unique_word_count', 'sentence_count', 'avg_word_length']].mean()
    return comparison_metrics

def plot_most_common_words(df, num_words=30):
    """Plot most common words for each label."""
    def plot_words(text_series, num_words, title):
        all_text = ' '.join(text_series).lower()
        words = all_text.split()
        word_freq = Counter(words)
        common_words = word_freq.most_common(num_words)

        plt.figure(figsize=(15, 6))
        sns.barplot(x=[word for word, freq in common_words], y=[freq for word, freq in common_words])
        plt.title(title)
        plt.xticks(rotation=45)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.show()
    
    plot_words(df[df['label'] == 0]['text'], num_words, "Most Common Words in Student Essays")
    plot_words(df[df['label'] == 1]['text'], num_words, "Most Common Words in LLM-generated Essays")

def plot_top_keywords(df, column='text', num_keywords=10):
    """Plot top keywords for each label."""
    def plot_keywords(data, column, num_keywords, label):
        vec = CountVectorizer(stop_words='english').fit(data[column])
        bag_of_words = vec.transform(data[column])
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        top_words = words_freq[:num_keywords]
        
        top_df = pd.DataFrame(top_words, columns=['Keyword', 'Frequency'])
        plt.figure(figsize=(10,6))
        sns.barplot(x='Frequency', y='Keyword', data=top_df)
        plt.title(f'Top {num_keywords} Keywords for label {label}')
        plt.show()
    
    plot_keywords(df, column, num_keywords, 0)
    plot_keywords(df, column, num_keywords, 1)

def analyze_ngrams(df, max_gram=5):
    """Analyze and visualize n-grams for each label."""
    def get_unique_ngrams(df, n):
        unique_ngrams = set()
        ngram_freq = {}
        for text in df['text']:
            words = text.lower().split()
            words = [word.strip(string.punctuation) for word in words]
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            unique_ngrams.update(set(ngrams))
            for ngram in ngrams:
                ngram_freq[ngram] = ngram_freq.get(ngram, 0) + 1
        return unique_ngrams, ngram_freq
    
    def display_dic(dic, columns, n):
        sorted_dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(sorted_dic, columns=columns)
        print(df.head(n))
        return df
    
    def show_unique_ngrams(n):
        unique_ngrams_ai, ngram_freq_ai = get_unique_ngrams(df[df['label'] == 1], n)
        unique_ngrams_human, ngram_freq_human = get_unique_ngrams(df[df['label'] == 0], n)

        labels = ['AI', 'Human']
        values = [len(unique_ngrams_ai), len(unique_ngrams_human)]
        plt.bar(labels, values, color=["coral", "lightgreen"])
        plt.title(f"Unique {n}-grams used by AI and human")
        plt.ylabel("n-grams")
        plt.show()

        ngrams_only_ai = set(unique_ngrams_ai) - set(unique_ngrams_human)
        ngram_freq_only_ai = {key: ngram_freq_ai[key] for key in ngram_freq_ai.keys() if key in ngrams_only_ai}
        ngrams_only_human = set(unique_ngrams_human) - set(unique_ngrams_ai)
        ngram_freq_only_human = {key: ngram_freq_human[key] for key in ngram_freq_human.keys() if key in ngrams_only_human}

        print(f"Top 10 {n}-grams only used by ai:")
        top_ngrams_ai = display_dic(ngram_freq_only_ai, [f"{n}-gram", "freq"], 10)
        print(f"\nTop 10 {n}-grams only used by human:")
        top_ngrams_human = display_dic(ngram_freq_only_human, [f"{n}-gram", "freq"], 10)
        return top_ngrams_ai, top_ngrams_human
    
    for n in range(1, max_gram + 1):
        show_unique_ngrams(n)
        print("\n\n")

def generate_wordclouds(df):
    """Generate word clouds for each label."""
    stop = set(nltk.corpus.stopwords.words('english'))
    
    def show_wordcloud(data):
        wordcloud = WordCloud(
            background_color='white',
            stopwords=stop,
            max_words=100,
            max_font_size=30,
            scale=3,
            random_state=1).generate(str(data))
        
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(wordcloud)
        plt.show()
    
    show_wordcloud(df[df['label'] == 1]['text'])
    show_wordcloud(df[df['label'] == 0]['text'])

def analyze_sentiment(df):
    """Analyze sentiment polarity for each label."""
    def polarity(text):
        return TextBlob(text).sentiment.polarity
    
    def sentiment(x):
        if x < -0.05:
            return 'neg'
        elif x > 0.05:
            return 'pos'
        else:
            return 'neu'
    
    data_human = df[df['label'] == 0].copy()
    data_LLM = df[df['label'] == 1].copy()
    
    data_human['polarity_score'] = data_human['text'].apply(polarity)
    data_LLM['polarity_score'] = data_LLM['text'].apply(polarity)
    
    # Plot histograms
    data_human['polarity_score'].hist()
    plt.title('Polarity Scores for Human-written Text')
    plt.show()
    
    data_LLM['polarity_score'].hist()
    plt.title('Polarity Scores for LLM-generated Text')
    plt.show()
    
    # Categorize sentiment
    data_human['polarity'] = data_human['polarity_score'].map(sentiment)
    data_LLM['polarity'] = data_LLM['polarity_score'].map(sentiment)
    
    # Plot sentiment distribution
    human_counts = data_human['polarity'].value_counts()/len(data_human)
    llm_counts = data_LLM['polarity'].value_counts()/len(data_LLM)
    
    combined_counts = pd.DataFrame({
        'Human': human_counts,
        'LLM': llm_counts
    }).fillna(0)
    
    combined_counts.plot(kind='bar', figsize=(10, 6), color=['blue', 'orange'])
    plt.xlabel('Polarity Classification')
    plt.ylabel('Percentage')
    plt.title('Polarity Distribution of Human vs LLM Data')
    plt.xticks(rotation=0)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()

def analyze_number_usage(df):
    """Analyze usage of numerals vs. number words."""
    numeral_pattern = re.compile(r'\b\d+\b')
    word_number_pattern = re.compile(
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b', re.IGNORECASE)
    
    def count_digit_styles(text, numeral_regex, word_number_regex):
        numeral_matches = len(re.findall(numeral_regex, text))
        word_number_matches = len(re.findall(word_number_regex, text))
        return numeral_matches, word_number_matches
    
    df = df.copy()
    df[['numerals', 'number_words']] = df['text'].apply(
        lambda x: count_digit_styles(x, numeral_pattern, word_number_pattern)).apply(pd.Series)
    
    student_essays = df[df['label'] == 0]
    llm_essays = df[df['label'] == 1]
    
    student_numerals = student_essays['numerals'].sum()
    student_number_words = student_essays['number_words'].sum()
    llm_numerals = llm_essays['numerals'].sum()
    llm_number_words = llm_essays['number_words'].sum()
    
    print(f"Student Essays - Numerals: {student_numerals}, Number Words: {student_number_words}")
    print(f"LLM Essays - Numerals: {llm_numerals}, Number Words: {llm_number_words}")
    
    avg_student_numerals = student_numerals / len(student_essays)
    avg_student_number_words = student_number_words / len(student_essays)
    avg_llm_numerals = llm_numerals / len(llm_essays) if len(llm_essays) > 0 else 0
    avg_llm_number_words = llm_number_words / len(llm_essays) if len(llm_essays) > 0 else 0
    
    print(f"\nAverage Numerals per Student Essay: {avg_student_numerals:.2f}")
    print(f"Average Number Words per Student Essay: {avg_student_number_words:.2f}")
    print(f"\nAverage Numerals per LLM Essay: {avg_llm_numerals:.2f}")
    print(f"Average Number Words per LLM Essay: {avg_llm_number_words:.2f}")


def analyze_readability(df):
    """Analyze various readability metrics with proper scaling."""
    data_human = df[df['label'] == 0].copy()
    data_LLM = df[df['label'] == 1].copy()

    def plot_metric(data, title, xlabel, bins=50, xlim=None):
        """Helper function to plot metrics with consistent formatting."""
        plt.figure(figsize=(10, 6))
        data.hist(bins=bins, grid=False, alpha=0.7)
        plt.title(title, fontsize=15)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        if xlim:
            plt.xlim(xlim)
        plt.show()

    # Flesch Reading Ease (standard scale: 0-100)
    human_fre = data_human['text'].apply(lambda x: max(0, min(flesch_reading_ease(x), 100)))
    llm_fre = data_LLM['text'].apply(lambda x: max(0, min(flesch_reading_ease(x), 100)))

    plot_metric(human_fre,
                'Flesch Reading Ease for Human-written Text',
                'Flesch Reading Ease (0-100 scale)',
                bins=200, xlim=(0, 100))

    plot_metric(llm_fre,
                'Flesch Reading Ease for LLM-generated Text',
                'Flesch Reading Ease (0-100 scale)',
                bins=200, xlim=(0, 100))

    # Flesch Kincaid Grade Level (standard scale: 0-20)
    human_fk = data_human['text'].apply(lambda x: max(0, min(flesch_kincaid_grade(x), 20)))
    llm_fk = data_LLM['text'].apply(lambda x: max(0, min(flesch_kincaid_grade(x), 20)))

    plot_metric(human_fk,
                'Flesch-Kincaid Grade Level for Human-written Text',
                'Grade Level (0-20 scale)',
                bins=200, xlim=(0, 20))

    plot_metric(llm_fk,
                'Flesch-Kincaid Grade Level for LLM-generated Text',
                'Grade Level (0-20 scale)',
                bins=200, xlim=(0, 20))

    # Dale-Chall Readability Score (standard scale: 0-10)
    human_dc = data_human['text'].apply(lambda x: max(0, min(dale_chall_readability_score(x), 10)))
    llm_dc = data_LLM['text'].apply(lambda x: max(0, min(dale_chall_readability_score(x), 10)))

    plot_metric(human_dc,
                'Dale-Chall Readability Score for Human-written Text',
                'Readability Score (0-10 scale)',
                bins=200, xlim=(0, 10))

    plot_metric(llm_dc,
                'Dale-Chall Readability Score for LLM-generated Text',
                'Readability Score (0-10 scale)',
                bins=200, xlim=(0, 10))

    # Difficult Words (raw counts, no scaling)
    human_dw = data_human['text'].apply(difficult_words)
    llm_dw = data_LLM['text'].apply(difficult_words)

    plot_metric(human_dw,
                'Difficult Words for Human-written Text',
                'Number of Difficult Words',
                bins=50, xlim=(0, 500))

    plot_metric(llm_dw,
                'Difficult Words for LLM-generated Text',
                'Number of Difficult Words',
                bins=50, xlim=(0, 500))

def main():
    split_raw_data_to_csv_dis_lora()
    # Load data
    essays_path = './data/data_merged_original_yuke.csv'
    train_df, test_df = load_data(essays_path)
    
    # Clean data
    # train_df = clean_data(train_df)
    
    # Perform analyses
    analyze_text_length(train_df)
    text_metrics = calculate_text_metrics(train_df)
    plot_most_common_words(train_df)
    plot_top_keywords(train_df)
    analyze_ngrams(train_df)
    generate_wordclouds(train_df)
    analyze_sentiment(train_df)
    analyze_number_usage(train_df)
    analyze_readability(train_df)

if __name__ == "__main__":
    main()