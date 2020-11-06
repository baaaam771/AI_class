import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(train_data, train_label), (test_data, test_label) = fashion_mnist.load_data()

train_data.shape, train_label.shape

plt.imshow(train_data[0])

train_label[0]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_data = train_data.reshape((60000, 28 * 28))
# 6만개
test_data = test_data.reshape((10000, 28*28))
# DNN 입력값에 reshape

train_data.shape, test_data.shape


# 모델 설계
# model=model.Sequential()
# model.add

model = models.Sequential([
    layers.Dense(units=512, activation='relu', input_shape=(28 * 28, )),
    layers.Dense(units=10, activation='softmax')
])

# 모델 구성
model.summary()

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 학습 후 결과 값 저장
history = model.fit(x=train_data,
                    y=train_label,
                    epochs=30,
                    batch_size=128,
                    validation_split=0.2
                    )

history.history


# 성능지표추출
accuracy = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# 차트 x축
epcohs = range(1, len(loss) + 1)

# loss를 시각화
plt.plot(epcohs, loss, 'bo', label='training loss')
plt.plot(epcohs, val_loss, 'r', label='validation loss')
plt.title('training and validation loss')
plt.legend()

# acc 시각화
plt.plot(epcohs, accuracy, 'bo', label='training acc')
plt.plot(epcohs, val_acc, 'r', label='validation acc')
plt.title('training and validation acc')
plt.legend()

# 모델 성능 평가
model.evaluate(x=test_data, y=test_label)

plt.imshow(test_data[10].reshape(28, 28))

print(class_names[test_label[10]])

# 학습된 모델에 10번째 데이터를 넣었을 때, 나오는 값을 확인
result = model.predict(test_data[10].reshape((1, 784)))

result

# 결과값 확인
result = np.argmax(result[0])
class_names[result]
