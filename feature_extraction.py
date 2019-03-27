import json_lines
import pandas as pd

truth = pd.read_json("truth.json")
instances = pd.read_json("instances.json")

instances['class'] = truth['truthClass']

# id postText postTimestamp postMedia targetTitle targetDescription targetKeywords targetParagraphs targetCaptions

features = pd.DataFrame(columns=['numCharPostTitle', 'numCharArticleTitle', 'numCharArticleDescr', 'numCharArticleKeywords',
                                 'numCharArticleCaption', 'numCharArticleParagraph', 'diffPostTitleArticleTitle',
                                 'diffPostTitleArticleDesc', 'diffPostTitleArticleKeywords', 'diffPostTitleArticleParagraph',
                                 'diffPostTitleArticleCaption', 'diffPostTitlePostImage', 'diffArticleTitleArticleDesc',
                                 'diffArticleTitleArticleKeywords', 'diffArticleTitleArticleParagraph', 'diffArticleTitleArticleCaption',
                                 'diffArticleTitlePostImage', 'diffArticleDescArticleKeywords', 'diffArticleDescArticleParagraph',
                                 'diffArticleDescArticleCaption', 'diffArticleDescPostImage', 'diffArticleKeywordsArticleParagraph',
                                 'diffArticleKeywordsArticleCaption', 'diffArticleKeywordsPostImage', 'diffArticleParagraphArticleCaption',
                                 'diffArticleParagraphPostImage', 'diffArticleCaptionPostImage', 'ratioPostTitleArticleTitle',
                                 'ratioPostTitleArticleDesc', 'ratioPostTitleArticleKeywords', 'ratioPostTitleArticleParagraph',
                                 'ratioPostTitleArticleCaption', 'ratioPostTitlePostImage', 'ratioArticleTitleArticleDesc',
                                 'ratioArticleTitleArticleKeywords', 'ratioArticleTitleArticleParagraph', 'ratioArticleTitleArticleCaption',
                                 'ratioArticleTitlePostImage', 'ratioArticleDescArticleKeywords', 'ratioArticleDescArticleParagraph',
                                 'ratioArticleDescArticleCaption', 'ratioArticleDescPostImage', 'ratioArticleKeywordsArticleParagraph',
                                 'ratioArticleKeywordsArticleCaption', 'ratioArticleKeywordsPostImage', 'ratioArticleParagraphArticleCaption',
                                 'ratioArticleParagraphPostImage', 'ratioArticleCaptionPostImage'])

for index, row in instances.iterrows():
    ### NUMBER OF CHARACTERS
    lenPostTitle = len(row['postText'][0])
    lenArtTitle = len(row['targetTitle'])
    lenArtDesc = len(row['targetDescription'])
    lenArtKeywords = len(row['targetKeywords'])
    print(len(row['postText'][0].split()))
    features_dict = {}
    features_dict['numCharPostTitle'] = [lenPostTitle]
    # MISSING THE NUMBER OF CHARACTERS FROM POST'S IMAGE
    features_dict['numCharArticleTitle'] = [lenArtTitle]
    features_dict['numCharArticleDescr'] = []
    features_dict['numCharArticleKeywords'] = [lenArtKeywords]

    lenArtCap = 0
    for caption in row['targetCaptions']:
        lenArtCap = lenArtCap + len(caption)
    features_dict['numCharArticleCaption'] = [lenArtCap]

    lenArtPar = 0
    for paragraph in row['targetParagraphs']:
        lenArtPar = lenArtPar + len(paragraph)
    features_dict['numCharArticleParagraph'] = [lenArtPar]

    ### DIFFERENCE IN CHARACTERS - STILL TO DO THE CHECK OF THE ZERO CHARACTER CASE
    features_dict['diffPostTitleArticleTitle'] = [abs(lenPostTitle-lenArtTitle)]
    features_dict['diffPostTitleArticleDesc'] = [abs(lenPostTitle-lenArtDesc)]
    features_dict['diffPostTitleArticleKeywords'] = [abs(lenPostTitle-lenArtKeywords)]
    features_dict['diffPostTitleArticleParagraph'] = [abs(lenPostTitle-lenArtPar)]
    features_dict['diffPostTitleArticleCaption'] = [abs(lenPostTitle-lenArtCap)]
    # STILL MISSING features_dict['diffPostTitlePostImage'] = [abs(lenPostTitle-lenPostImage)]
    features_dict['diffArticleTitleArticleDesc'] = [abs(lenArtTitle-lenArtDesc)]
    features_dict['diffArticleTitleArticleKeywords'] = [abs(lenArtTitle-lenArtKeywords)]
    features_dict['diffArticleTitleArticleParagraph'] = [abs(lenArtTitle-lenArtPar)]
    features_dict['diffArticleTitleArticleCaption'] = [abs(lenArtTitle-lenArtCap)]
    # STILL MISSING features_dict['diffArticleTitlePostImage'] = [abs(lenArtTitle-lenPostImage)]
    features_dict['diffArticleDescArticleParagraph'] = [abs(lenArtDesc-lenArtPar)]
    features_dict['diffArticleDescArticleKeywords'] = [abs(lenArtDesc-lenArtKeywords)]
    features_dict['diffArticleDescArticleCaption'] = [abs(lenArtDesc-lenArtCap)]
    # STILL MISSING features_dict['diffArticleDescPostImage'] = [abs(lenArtDesc-lenPostImage)]
    features_dict['diffArticleKeywordsArticleParagraph'] = [abs(lenArtKeywords-lenArtPar)]
    features_dict['diffArticleKeywordsArticleCaption'] = [abs(lenArtKeywords-lenArtCap)]
    # STILL MISSING features_dict['diffArticleKeywordsPostImage'] = [abs(lenArtKeywords-lenPostImage)]
    features_dict['diffArticleParagraphArticleCaption'] = [abs(lenArtPar-lenArtCap)]
    # STILL MISSING features_dict['diffArticleParagraphPostImage'] = [abs(lenArtPar-lenPostImage)]
    # STILL MISSING features_dict['diffArticleCaptionPostImage'] = [abs(lenArtCap-lenPostImage)]


    ### RATIO IN CHARACTERS - STILL TO DO THE CHECK OF THE ZERO CHARACTER CASE
    features_dict['ratioPostTitleArticleTitle'] = [1/abs(lenPostTitle/lenArtTitle)]
    features_dict['ratioPostTitleArticleDesc'] = [1/abs(lenPostTitle/lenArtDesc)]
    features_dict['ratioPostTitleArticleKeywords'] = [1/abs(lenPostTitle/lenArtKeywords)]
    features_dict['ratioPostTitleArticleParagraph'] = [1/abs(lenPostTitle/lenArtPar)]
    features_dict['ratioPostTitleArticleCaption'] = [1/abs(lenPostTitle/lenArtCap)]
    # STILL MISSING features_dict['ratioPostTitlePostImage'] = [1/abs(lenPostTitle/lenPostImage)]
    features_dict['ratioArticleTitleArticleDesc'] = [1/abs(lenArtTitle/lenArtDesc)]
    features_dict['ratioArticleTitleArticleKeywords'] = [1/abs(lenArtTitle/lenArtKeywords)]
    features_dict['ratioArticleTitleArticleParagraph'] = [1/abs(lenArtTitle/lenArtPar)]
    features_dict['ratioArticleTitleArticleCaption'] = [1/abs(lenArtTitle/lenArtCap)]
    # STILL MISSING features_dict['diffArticleTitlePostImage'] = [1/abs(lenArtTitle-lenPostImage)]
    features_dict['ratioArticleDescArticleParagraph'] = [1/abs(lenArtDesc/lenArtPar)]
    features_dict['ratioArticleDescArticleKeywords'] = [1/abs(lenArtDesc/lenArtKeywords)]
    features_dict['ratioArticleDescArticleCaption'] = [1/abs(lenArtDesc/lenArtCap)]
    # STILL MISSING features_dict['ratioArticleDescPostImage'] = [1/abs(lenArtDesc/lenPostImage)]
    features_dict['ratioArticleKeywordsArticleParagraph'] = [1/abs(lenArtKeywords/lenArtPar)]
    features_dict['ratioArticleKeywordsArticleCaption'] = [1/abs(lenArtKeywords/lenArtCap)]
    # STILL MISSING features_dict['ratioArticleKeywordsPostImage'] = [1/abs(lenArtKeywords/lenPostImage)]
    features_dict['ratioArticleParagraphArticleCaption'] = [1/abs(lenArtPar/lenArtCap)]
    # STILL MISSING features_dict['ratioArticleParagraphPostImage'] = [1/abs(lenArtPar/lenPostImage)]
    # features_dict['ratioArticleCaptionPostImage'] = [1/abs(lenArtCap/lenPostImage)]


    ### NUMBER OF WORDS


    features = features.append(pd.DataFrame.from_dict(features_dict), ignore_index=True)

print(features)
