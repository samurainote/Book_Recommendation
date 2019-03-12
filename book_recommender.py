
import sklearn
from sklearn.decomposition import TruncatedSVD

def engine(rating_data, book_name, top_n=10):

    SVD = TruncatedSVD(n_components=12, random_state=17)
    matrix_factorization = SVD.fit_transform(rating_data)
    corr = np.corrcoef(matrix_factorization)

    title = popular_books_pivot2.index
    book = list(title).index(book_name)
    book_corr = corr[book]
    top10_index = book_corr.argsort()[::-1][-11:-1]

    top10_book = []
    for index in top10_index.tolist():
        top_book_name = list(popular_books_pivot2.index)[index]
        top10_book.append(top_book_name)

    return top10_book
