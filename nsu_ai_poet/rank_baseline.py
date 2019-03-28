from nltk.stem.snowball import RussianStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from scipy.sparse import hstack


class BaselineRanking(BaseEstimator, ClassifierMixin):

    """Класс для ранджирования стихов по темам. Использует преобразование строки-темы и строки-стиха + логистическая
    регрессия на три класса"""

    def __init__(self):
        """
        vect_theme - векторизатор для строк-тем
        vect_poem - векторизатор для строк-стихов
        lin_model - обученная модель логрегрессии
        """
        self.vect_theme = None
        self.vect_poem = None
        self.lin_model = None
        self.stemmer = RussianStemmer(True)
        self.stop_w = stopwords.words('russian')

    @staticmethod
    def preprocess_text(text, stemmer, stop):
        """Токенизация, чистка и стэмминг текстов"""
        return ' '.join([stemmer.stem(i) for i in word_tokenize(text.lower()) if i not in stop])

    def fit(self, X_theme, X_poems, y, max_theme_feat=3000, max_poem_feat=5000, ngram_range=(3, 4)):
        """Обучаем ранжирование. Чистим все тексты, стеммим и токенизируем, обучаем Tf-idf отдельно для тем и стихов.
        Склеиваем вектора, обучаем логрегрессию.
        :param X_theme: список строк-тем для стихов
        :param X_poems: список строк-стихов
        :param y: список оценок от 1 до 5
        :param max_poem_feat: максимальное число признаков для TfIdf стихов
        :param max_theme_feat: максимельное число признаков для Tfidf тем
        :param ngram_range: размах симсвольных н-грам для обучения
        :return self: обученная модель"""
        tem_norm = [self.preprocess_text(text, self.stemmer, self.stop_w) for text in X_theme]
        poems_norm = [self.preprocess_text(text, self.stemmer, self.stop_w) for text in X_poems]

        self.vect_theme = TfidfVectorizer(analyzer='char', max_features=max_theme_feat, ngram_range=ngram_range)
        self.vect_poem = TfidfVectorizer(max_features=max_poem_feat, analyzer='char', ngram_range=ngram_range)
        X_tem = self.vect_theme.fit_transform(tem_norm)
        X_poem = self.vect_poem.fit_transform(poems_norm)

        X = hstack([X_tem, X_poem])

        #self.lin_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self.lin_model = MLPRegressor(hidden_layer_sizes=(1000,), early_stopping=True, verbose=True, alpha=0.005)
        self.lin_model.fit(X, y)

        return self

    def predict(self, X_theme, X_poems):
        """Предсказание. Чистим, стеммим и преобразуем темы и стихи, склеиваем, запускаем в логрегрессию.
        :param X_theme: список строк-тем стихов
        :param X_poems: список строк-стихов
        :return список оценок от 1 до 5, соответствие стиха теме"""
        check_is_fitted(self, ['vect_theme', 'vect_poem', 'lin_model'])
        tem_norm = [self.preprocess_text(text, self.stemmer, self.stop_w) for text in X_theme]
        poems_norm = [self.preprocess_text(text, self.stemmer, self.stop_w) for text in X_poems]

        X_tem = self.vect_theme.transform(tem_norm)
        X_poem = self.vect_poem.transform(poems_norm)

        X = hstack([X_tem, X_poem])

        return self.lin_model.predict(X)

    def __getstate__(self):
        return {'vect_theme': self.vect_theme, 'vect_poem': self.vect_poem, 'linear_model': self.lin_model}

    def __setstate__(self, state):
        self.vect_poem = state['vect_poem']
        self.vect_theme = state['vect_theme']
        self.lin_model = state['linear_model']
        self.stemmer = RussianStemmer(True)
        self.stop_w = stopwords.words('russian')

