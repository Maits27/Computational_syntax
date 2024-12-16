# Computational Syntax - HMM PoS Tagger

This project is made by: Adrian Cuadrón, Aimar Sagasti and Maitane Urruela

## Overview
This project involves implementing a Hidden Markov Model (HMM) for Part-of-Speech (PoS) tagging, following the model discussed in the subject of Computational Syntax. This project will involve the development of two HMMs, one for each language: English and Spanish. These models will be trained using datasets from [Universal Dependencies](https://universaldependencies.org/), with a focus on formal contexts such as legal texts, news articles, Wikipedia, and other similar domains. 

The performance of both models will be evaluated in an **in-domain** context, ensuring they accurately handle the formal language structures of these specialized datasets. In addition, **out-of-domain** data will be collected from social media platforms, such as Twitter, to test the models' robustness and effectiveness in processing informal, colloquial language. This will allow for a comprehensive evaluation of each model's ability to adapt to different language environments, providing insights into how well they generalize across diverse contexts.

## Datasets used:

### In-Domain Datasets:

For the in-domain part, the following datasets have been used:

* English: [UD_English-ParTUT](https://github.com/UniversalDependencies/UD_English-ParTUT/tree/master), a conversion of a multilingual parallel treebank developed at the University of Turin, and consisting of a variety of text genres, including talks, legal texts and Wikipedia articles, among others. It contains 2.090 sentences, with a total of 49.648 words, divided in train, dev and test.
* Spanish: [AnCora](https://github.com/UniversalDependencies/UD_Spanish-AnCora/blob/master/), a conversion of the AnCora corpus to Universal Dependencies guidelines consisting only on news-domain corpora. However, the amount of data is quite bigger in comparison to the English corpora, having more than 568K words.

### Out-of-Domain Datasets:

In the case of out-of-domain field, for both languages we have selected Twitter data:

* English: [Tweebank](https://github.com/Oneplus/Tweebank/tree/dev), a collection of English tweets annotated in Universal Dependencies.
* Spanish: [xLiMe Twitter Corpus](https://github.com/lrei/xlime_twitter_corpus/tree/master), annotated with PoS tagging. Since the data was annotated in a txt only using the word and the tag, we had to put it in the Universal Dependencies format ourselves.

## Structure of the Project
### Implementation of the HMM PoS Tagger
The HMM PoS tagger will be implemented in Python. The model will be trained on annotated datasets and will tag unseen text according to the PoS tag set defined by the datasets.

The core of the model follows the HMM structure discussed in class, which includes:

* Transition Probabilities: Probability of moving from one tag to another.
* Emission Probabilities: Probability of a word being emitted given a specific tag.
* Viterbi Algorithm: Used to determine the most likely sequence of tags for a given sequence of words.

**IMPORTANT:** All the code related to the implementation of the HMM is located in the [*main.py*](https://github.com/Maits27/Computational_syntax/blob/main/main.py) file in the root directory of this project. All the probabilities are in logarithmic scale to avoid underflow issues.

### In-Domain Experiments

The model will be evaluated on the two datasets from the Universal Dependencies (UD) project. 

* Training: The model will be trained on the train files of each dataset and evaluated with the dev file.
* Testing: The model will be tested on the test file of the dataset, making use of the rest of the data to train the model.
* Evaluation Metrics: The accuracy of the model will be reported, with a detailed error analysis provided. This will include an examination of the types of errors made.

**IMPORTANT:** All the error analysis of the in-domain part is located in the Jupiter Notebook [*resultAnalysis.ipynb*](https://github.com/Maits27/Computational_syntax/blob/main/resultAnalysis.ipynb) in the root directory of this project.

### Out-of-Domain Results

After the in-domain experiments, the model will be tested on out-of-domain data from Twitter. 
* Evaluation: The evaluation will be made using the fully trained HMM for each language.
* Domain Shift Analysis: It will be analyzed how robust the model is to domain shift. 

**IMPORTANT:** All the error analysis of the out-of-domain part is located in the Jupiter Notebook [*OD_resultAnalysis.ipynb*](https://github.com/Maits27/Computational_syntax/blob/main/OD_resultAnalysis.ipynb) in the root directory of this project.
