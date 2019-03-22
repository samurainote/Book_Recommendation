# Collaborative Filtering with SVD for Book Recommendation
## 強調フィルタリングと特異値分解を用いた本の推薦エンジン
![](https://buildingrecommenders.files.wordpress.com/2015/11/matrix-factorisation.png?w=900)

## Introduction

#### Machine Learning in Recommend System

Collaborative filtering uses both purchase and purchase data from one target person and products checked by another person, based on the purchase pattern, the similarity between people or co-occurrence between products. It is a method that can present personalized products by analyzing in association analysis (correlation analysis) and correlating the action history of the target person. 

The premise is that “the evaluation of people who are similar to you will be similar to your evaluation. Therefore, the difference is that you want products that you do not have, but people who are similar to you have. There is no such thing! Therefore, it is basic to look for people who are similar to you among many customers, and to recommend the ones that those people have that you do not have.

## Technical Preferences

| Title | Detail |
|:-----------:|:------------------------------------------------|
| Environment | MacOS Mojave 10.14.3 |
| Language | Python |
| Library | scikit-learn, Numpy, matplotlib, Pandas, Seaborn |
| Dataset | [Book-crossing dataset ](http://www2.informatik.uni-freiburg.de/~cziegler/BX/) |
| Algorithm | Collaborative Filtering, Singular Value Decomposition |

## References

- [Recommender System using Singular Value Decomposition](http://rstudio-pubs-static.s3.amazonaws.com/335300_11d40bf12d8940f78d9661b3c63150dc.html)
- [Singular Value decomposition (SVD) in recommender systems](https://medium.com/@m_n_malaeb/singular-value-decomposition-svd-in-recommender-systems-for-non-math-statistics-programming-4a622de653e9)
- [Various Implementations of Collaborative Filtering](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0)
- [Collaborative Filtering based Recommendation Systems exemplified..](https://towardsdatascience.com/collaborative-filtering-based-recommendation-systems-exemplified-ecbffe1c20b1)
- [Machine Learning. Explanation of Collaborative Filtering vs Content Based Filtering.](https://codeburst.io/explanation-of-recommender-systems-in-information-retrieval-13077e1d916c)
