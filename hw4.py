from typing import List
import math 

class CountVectorizer():
    """
    Класс "мешок слов"\n
    Атрибуты: \n
        vocabulary: словарик документа\n
        number_of_words: количество слов в документе \n
        count_matrix: матрица слов по предложениям\n
    Методы: \n
        get_feature_names(): возвращает словарик (работает после fit_transform)\n
        fit_transform(text): возвращает матрицу слов в предложении (на вход подаётся текст)\n
        метод заполняет все атрибуты\n
    """
    def __init__(self):
        self.vocabulary = []
        self.number_of_words = 0
        self.count_matrix = []

    def get_feature_names(self):
        """
        возвращает словарик документа (работает после fit_transform)
        """
        return self.vocabulary
        
    def fit_transform(self, text: List[str]) -> List[dict]:
        """
        возвращает матрицу слов в предложении (на вход подаётся текст), \n
        метод заполняет все атрибуты класса CountVectorizer
        """
        list_of_dictionaries=[]
        text_splited=text.copy()
        for isentence, sentence in enumerate(text_splited):
            sentence_splited = sentence.split()
            text_splited[isentence] = [words.lower() for words in sentence_splited]
        for sentence in text_splited:
            dictionary = {word: 0 for sentence in text_splited for word in sentence}
            self.vocabulary = list(dictionary.keys())
            for word in sentence:
                dictionary[word] += 1
                self.number_of_words += 1
            list_of_dictionaries.append(list(dictionary.values()))
            self.count_matrix = list_of_dictionaries
        return list_of_dictionaries


def tf_transform(count_matrix: List[List[int]]) -> List[List[float]]:
    """
    На вход подаётся матрица, на выходе tf преобразование матрицы
    """
    list_of_tf_transform=[]
    for count_sentence in count_matrix:
        list_of_tf_transform.append([round(i/sum(count_sentence), 3) for i in count_sentence])
    return list_of_tf_transform


def idf_transform(count_matrix: List[List[int]]) -> List[float]:
    """
    На вход подаётся матрица, на выходе idf преобразование матрицы
    """
    list_of_idf_transform = [0 for _ in range(len(count_matrix[0]))]
    for count_sentence in count_matrix:
        for iword, word in enumerate(count_sentence):
            if word > 0:
                list_of_idf_transform[iword]+=1
    list_of_idf_transform=[round(math.log((len(count_matrix)+1)/(i+1)) + 1, 3) for i in list_of_idf_transform]
    return list_of_idf_transform


class TfidfTransformer():
    """
    Класс преобразования матриц\n
    Атрибуты:\n
        _tf: term frequency\n
        _idf: inverse document-frequency\n
        _tfidf: перемноженные tf и idf матрицы\n
    Методы:\n
        tf_transform(count_matrix): на вход подаётся матрица, на выходе tf преобразование матрицы,\n
        заполняется атрибут _tf\n
        idf_transform(count_matrix): на вход подаётся матрица, на выходе idf преобразование матрицы,\n
        заполняется атрибут _idf\n
        fit_transform: перемножает tf и idf матрицы\n 
    """
    def __unit__(self):
        self._tf=[]
        self._idf=[]
        self._tfidf=[]

    def tf_transform(self, count_matrix: List[List[int]]) -> List[List[float]]:
        """
        на вход подаётся матрица, на выходе tf преобразование матрицы,\n
        заполняется атрибут _tf класса TfidfTransformer
        """
        list_of_tf_transform=[]
        for count_sentence in count_matrix:
            list_of_tf_transform.append([round(i/sum(count_sentence), 3) for i in count_sentence])
        self._tf=list_of_tf_transform
        return list_of_tf_transform

    def idf_transform(self, count_matrix: List[List[int]]) -> List[float]:
        """
        на вход подаётся матрица, на выходе idf преобразование матрицы,\n
        заполняется атрибут _idf класса TfidfTransformer
        """
        list_of_idf_transform = [0 for _ in range(len(count_matrix[0]))]
        for count_sentence in count_matrix:
            for iword, word in enumerate(count_sentence):
                if word > 0:
                    list_of_idf_transform[iword]+=1
        list_of_idf_transform=[round(math.log((len(count_matrix)+1)/(i+1)) + 1, 1) for i in list_of_idf_transform]
        self._idf = list_of_idf_transform
        return list_of_idf_transform

    def fit_transform(self, count_matrix: List[List[int]]) -> List[List[int]]:
        """
        перемножает tf и idf матрицы
        """        
        self.tf_transform(count_matrix)
        self.idf_transform(count_matrix)
        tfidf = []
        for row, _ in enumerate(count_matrix):
            document_tfidf = []
            for col, _ in enumerate(count_matrix[0]):
                document_tfidf.append(round(self._tf[row][col] * self._idf[col], 3))
            tfidf.append(document_tfidf)
        self._tfidf=tfidf
        return tfidf


class TfidfVectorizer(CountVectorizer):
    """
    Класс "мешок слов", который умеет в преобразование матриц\n
    Наследуется от обычного мешка слов и является композицией класса преобразования матриц tdidf
    Атрибуты: 
        vocabulary: словарик документа\n
        number_of_words: количество слов в документе \n
        count_matrix: матрица слов по предложениям\n
        tf_idf_transformer: instance класса преобразования матриц\n
    Методы:
        get_feature_names(): возвращает словарик (работает после fit_transform)\n
        fit_transform(text): возвращает TDIDF матрицу слов в предложении (на вход подаётся текст)\n
        метод заполняет все атрибуты\n
    """
    def __init__(self):
        super().__init__()
        self.tf_idf_transformer = TfidfTransformer()

    def fit_transform(self, corpus):
        matrix = super().fit_transform(corpus)
        tdidfmatrix = self.tf_idf_transformer.fit_transform(matrix)
        return tdidfmatrix


if __name__ == '__main__':
    task = 4 # 1 или 2 или 3 или 4 или 5 
    if task == 1:
        corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
        ]
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform(corpus)
        print(vectorizer.get_feature_names())
        #Out: ['crock', 'pot', 'pasta', 'never', 'boil', 'again', 'pomodoro',
        #_ 'fresh', 'ingredients', 'parmesan', 'to', 'taste']
        print(count_matrix)
        #Out: [[1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        #[0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    elif task == 2:
        count_matrix = [
            [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ]
        tf_matrix = tf_transform(count_matrix)
        print(tf_matrix)
        #Out: [[0.143, 0.143, 0.286, 0.143, 0.143, 0.143, 0, 0, 0, 0, 0, 0],
        #[0, 0, 0.143, 0, 0, 0, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143]]
    elif task == 3:
        count_matrix = [
            [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ]
        idf_matrix = idf_transform(count_matrix)
        print(idf_matrix)
        #Out: [1.4, 1.4, 1.0, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]
    elif task == 4:
        count_matrix = [
            [1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            ]
        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(count_matrix)
        print(tfidf_matrix)
        #Out: [[0.2, 0.2, 0.286, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
        #[0, 0, 0.143, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]
    elif task == 5:
        corpus = [
            'Crock Pot Pasta Never boil pasta again',
            'Pasta Pomodoro Fresh ingredients Parmesan to taste'
            ]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        print(vectorizer.get_feature_names())
        #Out: ['crock', 'pot', 'pasta', 'never', 'boil', 'again', 'pomodoro',
        #_ 'fresh', 'ingredients', 'parmesan', 'to', 'taste']
        print(tfidf_matrix)
        #Out: [[0.2, 0.2, 0.286, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
        #[0, 0, 0.143, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]]     
    else:
        print("Не было такого задания, исправь в начале main номер задания")   

