import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pickle

def trainer_data():

        # Пример данных
        X_train_texts = [

                #ГРИПП
            "головная_боль насморк чихание боль_в_горле температура", #1
            "насморк головная_боль чихание температура боль_в_горле", #2
            "температура головная_боль насморк чихание боль_в_горле", #3
            "чихание головная_боль боль_в_горле температура насморк", #4
            
                #КОРОНАВИРУС
            "насморк боль_в_груди чихание температура",
            "температура боль_в_груди чихание насморк ",
            "чихание температура насморк боль_в_груди ",
            
                #АНГИНА
            "температура боль_в_горле тошнота головокружение",
            "боль_в_горле температура головокружение тошнота",
            "головокружение температура тошнота боль_в_горле",
            "тошнота боль_в_горле головокружение температура",
            
                #ПРОСТУДА
            "головная_боль насморк чихание головокружение",
            "головокружение головная_боль чихание насморк",
            "чихание насморк головокружение головная_боль",
            
                #АЛЛЕРГИЯ
            "зуд головокружение температура",
            "головокружение зуд температура",
            "температура головокружение зуд"
        ]


        #------------------------------------------------------------------------------------------------------------
                                 # Преобразование текста в числовой формат с помощью Tokenizer
        #------------------------------------------------------------------------------------------------------------



        # Преобразование текста в числовой формат с помощью Tokenizer
        tokenizer = Tokenizer(filters = " ")
        tokenizer.fit_on_texts(X_train_texts)
        X_train_sequences = tokenizer.texts_to_sequences(X_train_texts)
        X_train = tokenizer.sequences_to_matrix(X_train_sequences, mode='binary')



        # Преобразование меток классов в многомерный массив
        y_train = ["Грипп",
                   "Грипп",
                   "Грипп",
                   "Грипп",
                   "Коронавирус",
                   "Коронавирус",
                   "Коронавирус",
                   "Ангина",
                   "Ангина",
                   "Ангина",
                   "Ангина",
                   "Простуда",
                   "Простуда",
                   "Простуда",
                   "Аллергия",
                   "Аллергия",
                   "Аллергия"]

        num_classes = len(set(y_train))
        label_to_index = {label: i for i, label in enumerate(set(y_train))}
        y_train_indices = [label_to_index[label] for label in y_train]
        Y_train = to_categorical(y_train_indices, num_classes)



        import pickle

        # Сохранение объекта Tokenizer в файл
        with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


        ### Сохранение словаря label_to_index с использованием pickle
        # Сохранение словаря label_to_index в файл
        with open('label_to_index.pickle', 'wb') as handle:
                pickle.dump(label_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #------------------------------------------------------------------------------------------------------------
                                                # Создание и компиляция модели
        #------------------------------------------------------------------------------------------------------------


        # Создание и компиляция модели
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # Первый скрытый слой
        model.add(Dense(64, activation='relu'))  # Второй скрытый слой
        model.add(Dense(num_classes, activation='softmax'))  # Выходной слой

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #binary_crossentropy


        #------------------------------------------------------------------------------------------------------------
                                                # Обучение модели
        #------------------------------------------------------------------------------------------------------------


        model.fit(X_train, Y_train, epochs=15, batch_size=32, validation_split=0.001)

        # Сохранение обученной модели
        model.save("trained_model.keras")

    
        #------------------------------------------------------------------------------------------------------------
                                               # Создание тестового набора данных
        #------------------------------------------------------------------------------------------------------------


        X_test_texts = ["боль_в_груди температура"]

        # Преобразование тестовых данных
        X_test_sequences = tokenizer.texts_to_sequences(X_test_texts)
        X_test = tokenizer.sequences_to_matrix(X_test_sequences, mode='binary')


        #------------------------------------------------------------------------------------------------------------
                                        # Оценка и использование модели
        #------------------------------------------------------------------------------------------------------------
                                        
        predictions = model.predict(X_test)
        predicted_labels = [list(label_to_index.keys())[list(label_to_index.values()).index(np.argmax(pred))] for pred in predictions]


        # Вывод результатов
        print("Предсказания:", predicted_labels)




def opredelenie_bolezni(input_data):

        from keras.models import load_model
        import pickle

        # Загрузка сохраненной модели
        loaded_model = load_model("trained_model.keras")

                # Загрузка объекта Tokenizer из файла
        with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

        # Загрузка словаря label_to_index из файла
        with open('label_to_index.pickle', 'rb') as handle:
                label_to_index = pickle.load(handle)

        sequences = tokenizer.texts_to_sequences(input_data)
        X = tokenizer.sequences_to_matrix(sequences, mode='binary')
        predictions = loaded_model.predict(X)
        predicted_labels = [list(label_to_index.keys())[list(label_to_index.values()).index(np.argmax(pred))] for pred in predictions]
        return predicted_labels


#trainer_data()
test_input = ["насморк чихание боль_в_горле температура"]
predicted_disease = opredelenie_bolezni(test_input)
print(test_input)
print("Определение болезни:", predicted_disease)

        











