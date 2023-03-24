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

### Built With
To process a text and obtain its vectors we need to preprocess it. To do this we implemented a set of techniques for text preprocessing based on [NLTK](https://www.nltk.org/) and [Spacy](https://spacy.io/).

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

