# Eclectic representation of text
<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<p style="text-align: justify;">The project is about how computing and linguistics are converging to equip computers with the ability to process and interpret complex sentence structures of human language, resulting in the emergence of areas like Text Mining and Natural Language Processing. Feature extraction is used to analyze language from a mathematical perspective to extract different text properties as a vector representation. Several feature extraction techniques are commonly used, depending on the purpose of the recognition task, and this project proposes a novel method called an eclectic representation that unifies different informative aspects to improve the effectiveness of recognition models.</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Description
To process a text and obtain its vectors we need to preprocess it. To do this we implemented a set of techniques for text preprocessing based on [NLTK](https://www.nltk.org/) and [Spacy](https://spacy.io/).
This stage may include:
* Cleaning

Cleaning undesirable symbols, exclamation points, question marks, hashtags, apostrophes, URLs, HTML tags, in some cases numbers.
* Tokenization
Tokenize the documents involves breaking down the phrases into words or n-grams, each of these parts is known as a token.
* Stop words
Some words that we usually use in almost all conversations do not carry much value for the purpose of a machine learning task, therefore the most convenient task is to remove them from the text corpus.
* Stemming 
Stemming is a natural language processing technique that reduces a word to its base or root form, called the "stem". This is done by removing common suffixes or prefixes from a word.
* Lemmatization 
Lemmatization is a natural language processing technique that involves reducing a word to its base or dictionary form, called the "lemma". Unlike stemming, lemmatization takes into account the context and the part of speech of the word to ensure that the resulting lemma is a valid word that represents the correct meaning of the original word.



This section describes the feature extraction techniques used to encode the text. 
It is focused on the three components of text: 
```
1. lexical
2. syntactic
3. semantic
```
The text needs 
For this purpose, we implemented the next feature extraction techniques:
* TF-IDF
* DOC2VEC
* BERT

Once we have 

