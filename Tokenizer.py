
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Предположим, что у вас есть список текстов
texts = [
    "Пример текста 1",
    "Некоторый другой текст для обучения модели",
    "Еще один пример текста"
]

# Создаем токенизатор и подгоняем его на текстах
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Преобразовываем текст в последовательности числовых значений
sequences = tokenizer.texts_to_sequences(texts)

print(sequences)
