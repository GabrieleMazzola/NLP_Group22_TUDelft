from gabry_dataset_parser import get_labeled_instances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from nltk.parse.corenlp import CoreNLPDependencyParser
from pycorenlp import StanfordCoreNLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv


import numpy as np
import matplotlib.pyplot as plt



labeled_instances = get_labeled_instances("./train_set/instances_converted.pickle",
                                          "./train_set/truth_converted.pickle")


clickbait_df = labeled_instances[(labeled_instances.truthClass == 'clickbait')]
no_clickbait_df = labeled_instances[(labeled_instances.truthClass == 'no-clickbait')]

print(clickbait_df.columns)

def get_slang_words_list():
    slang_data = []
    with open('slang_dict.doc', 'r') as exRtFile:
        exchReader = csv.reader(exRtFile, delimiter='`', quoting=csv.QUOTE_NONE)
        for row in exchReader:
            slang_data.append(row)

    slang_words = []
    for i, text in enumerate(slang_data):
        slang_words.append(text[0])
    return slang_words

################# STOPWRODS ANALYSIS #################
def stopwords_analysis():
    stopWords = set(stopwords.words('english'))

    stopwords_count_clickbait = 0
    total_words = 0

    #for i, text in enumerate(clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        text_post = text[0].lower()
        word_tokens = text_post.split()
        stopwords_in_sentence = [w for w in word_tokens if w in stopWords]

        stopwords_count_clickbait += len(stopwords_in_sentence)
        total_words += len(word_tokens)

    percentage_stopwords_clickbait = (stopwords_count_clickbait * 100) / total_words

    stopwords_count_nonclickbait = 0
    total_words = 0
    #for i, text in enumerate(no_clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        text_post = text[0].lower()
        word_tokens = text_post.split()
        stopwords_in_sentence = [w for w in word_tokens if w in stopWords]

        stopwords_count_nonclickbait += len(stopwords_in_sentence)
        total_words += len(word_tokens)

    percentage_stopwords_noclickbait = (stopwords_count_nonclickbait * 100) / total_words
    print(f"Stopwords clickbait: {percentage_stopwords_clickbait}")
    print(f"Stopwords non_clickbait: {percentage_stopwords_noclickbait}")

    # libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # set width of bar
    barWidth = 0.05

    # set height of bar
    bars1 = [percentage_stopwords_clickbait]
    bars2 = [percentage_stopwords_noclickbait]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, bars1, color='blue', width=barWidth, edgecolor='white', label='Clickbait')
    plt.bar(r2, bars2, color='red', width=barWidth, edgecolor='white', label='Non-clickbait')

    # Add xticks on the middle of the group bars
    plt.xlabel('Category', fontweight='bold')
    plt.ylabel('% stopwords', fontweight='bold')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    # Create legend & Show graphic
    plt.legend()
    # plt.show()



#######################SHORTENINGS IN HEADLINE##############################
def shortenings_headline():
    text_file = open("shortenings.txt", "r")
    shortenings = text_file.read().split('\n')
    print(shortenings)

    shortenings_count_clickbait = 0
    total_words = 0
    #for i, text in enumerate(clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        text_post = text[0].lower()
        word_tokens = text_post.split()
        shortenings_in_sentence = [w for w in word_tokens if w in shortenings]

        shortenings_count_clickbait += len(shortenings_in_sentence)
        total_words += len(word_tokens)
    percentage_shortenings_clickbait = (shortenings_count_clickbait * 100) / total_words


    shortenings_count_nonclickbait = 0
    total_words = 0
    #for i, text in enumerate(no_clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        text_post = text[0].lower()
        word_tokens = text_post.split()
        shortenings_in_sentence = [w for w in word_tokens if w in shortenings]

        shortenings_count_nonclickbait += len(shortenings_in_sentence)
        total_words += len(word_tokens)

    percentage_shortenings_noclickbait = (shortenings_count_nonclickbait * 100) / total_words
    print(f"Shortenings clickbait: {percentage_shortenings_clickbait}")
    print(f"Shortenings non-clickbait: {percentage_shortenings_noclickbait}")


#SYNTACTIC DEPENDENCIES

def syntactic_dependencies():
    # path = './stanford-parser-full-2018-10-17/stanford-parser.jar'
    # path_to_model = './stanford-english-corenlp-2018-10-05-models.jar'

    # dependency_parser = CoreNLPDependencyParser()
    # res = word_tokenize('I love dogs the most in the world')
    # res_parse = dependency_parser.raw_parse('I love dogs the most in the world')
    # print(res_parse)

    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
    nlp = StanfordCoreNLP('http://localhost:9000')
    text = 'I shot an elephant in my sleep'
    output = nlp.annotate(text, properties={'annotators': 'depparse', 'outputFormat': 'json'})
    print(output['sentences'][0].keys())
    print(output['sentences'][0]['basicDependencies'])


################SLANG####################
def slang_words_analysis():
    slang_words = get_slang_words_list()

    slang_count_clickbait = 0
    total_words = 0
    #for i, text in enumerate(clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        text_post = text[0].replace("RT", "")
        word_tokens = text_post.split()
        slang_in_sentence = [w for w in word_tokens if w in slang_words]

        slang_count_clickbait += len(slang_in_sentence)
        total_words += len(word_tokens)
    percentage_slang_clickbait = (slang_count_clickbait * 100) / total_words

    slang_count_nonclickbait = 0
    total_words = 0
    #for i, text in enumerate(no_clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        text_post = text[0].replace("RT", "")
        word_tokens = text_post.split()
        slang_in_sentence = [w for w in word_tokens if w in slang_words]

        slang_count_nonclickbait += len(slang_in_sentence)
        total_words += len(word_tokens)

    percentage_slang_noclickbait = (slang_count_nonclickbait * 100) / total_words
    print(f"Slang clickbait: {percentage_slang_clickbait}")
    print(f"Slang non-clickbait: {percentage_slang_noclickbait}")




################SENTIMENT + POLARITY####################
def sentiment_analysis():
    analyser = SentimentIntensityAnalyzer()

    polarity_neutral_clickbait = 0
    count = 0
    #for i, text in enumerate(clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        score = analyser.polarity_scores(text[0])
        polarity_neutral_clickbait += score['neu']
        count += 1

    print(f"Neutral clickbait: {polarity_neutral_clickbait/count}")

    polarity_neutral_noclickbait = 0
    count = 0
    #for i, text in enumerate(no_clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        score = analyser.polarity_scores(text[0])
        polarity_neutral_noclickbait += score['neu']
        count += 1

    print(f"Neutral non-clickbait: {polarity_neutral_noclickbait / count}")




################# STOPWRODS ANALYSIS #################
def stopwords_analysis_paragraphs():
    stopWords = set(stopwords.words('english'))

    stopwords_count_clickbait = 0
    total_words = 0

    normalized_stopwords_clickbait = []
    #for i, text in enumerate(clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):

        stopwords_count_clickbait = 0
        total_words = 0

        for par in text:
            text_post = par.lower()
            word_tokens = text_post.split()
            stopwords_in_sentence = [w for w in word_tokens if w in stopWords]

            stopwords_count_clickbait += len(stopwords_in_sentence)
            total_words += len(word_tokens)

        if total_words != 0:
            normalized_stopwords_clickbait.append(stopwords_count_clickbait/total_words)

    percentage_stopwords_clickbait = np.mean(normalized_stopwords_clickbait)

    normalized_stopwords_noclickbait = []
    #for i, text in enumerate(no_clickbait_df['postText']):
    for i, text in enumerate(no_clickbait_df['targetParagraphs']):

        stopwords_count_nonclickbait = 0
        total_words = 0

        for par in text:
            text_post = par.lower()
            word_tokens = text_post.split()
            stopwords_in_sentence = [w for w in word_tokens if w in stopWords]

            stopwords_count_nonclickbait += len(stopwords_in_sentence)
            total_words += len(word_tokens)

        if total_words != 0:
            normalized_stopwords_noclickbait.append(stopwords_count_nonclickbait/total_words)

    percentage_stopwords_noclickbait = np.mean(normalized_stopwords_noclickbait)

    print(f"Stopwords clickbait: {percentage_stopwords_clickbait}")
    print(f"Stopwords non_clickbait: {percentage_stopwords_noclickbait}")



def sentiment_analysis_paragraphs():
    analyser = SentimentIntensityAnalyzer()

    polarity_neutral_clickbait = 0
    count = 0
    #for i, text in enumerate(clickbait_df['postText']):
    for i, text in enumerate(clickbait_df['targetParagraphs']):
        polarity_neutral_par = 0
        count_par = 0
        for par in text:
            score = analyser.polarity_scores(par)
            polarity_neutral_par += score['neu']
            count_par += 1

        if count_par != 0:
            polarity_neutral_clickbait += polarity_neutral_par/count_par
            count +=1

    print(f"Neutral clickbait: {polarity_neutral_clickbait/count}")

    polarity_neutral_noclickbait = 0
    count = 0
    #for i, text in enumerate(no_clickbait_df['postText']):
    for i, text in enumerate(no_clickbait_df['targetParagraphs']):
        polarity_neutral_noclickbait_par = 0
        count_par = 0

        for par in text:
            score = analyser.polarity_scores(par)
            polarity_neutral_noclickbait_par += score['neu']
            count_par += 1

        if count_par != 0:
            polarity_neutral_noclickbait += polarity_neutral_noclickbait_par/count_par
            count += 1

    print(f"Neutral non-clickbait: {polarity_neutral_noclickbait / count}")


#stopwords_analysis_paragraphs()
sentiment_analysis_paragraphs()
#shortenings_headline()
#sentiment_analysis()
#slang_words_analysis()