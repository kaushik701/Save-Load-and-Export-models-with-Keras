#%%
import tensorflow as tf
import os
from tensorflow import keras
import h5py
print(tf.__version__)
# %%
folders = ['tmp', 'models', 'model_name', 'weights']
for folder in folders:
    if not os.path.isdir(folder):
        os.makedirs(folder)
print(os.listdir('.'))
# %%
def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', input_shape=(784,)))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='softmax'))
    model.add(keras.layers.Dense(128, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = create_model()
model.summary()
# %%
import numpy as np
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = np.reshape(train_images, (train_images.shape[0], 784))/255.0
test_images = np.reshape(test_images, (test_images.shape[0], 784))/255.0

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)
# %%
checkpoint_dir = 'weights/'
_ = model.fit(
    train_images, train_labels, validation_data = (test_images, test_labels),
    epochs=2, batch_size=512,
    callbacks=[keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir,'epoch_{epoch:02d}_accuracy_{val_acc:.4f}.h5'),
    monitor='val_acc', save_weights_only=True, save_best_only=True)]
)
#%%
os.listdir(checkpoint_dir)
#%%
model = create_model()
print(model.evaluate(test_images, test_labels, verbose=False))
model.load_weights('weights/epoch_02_accuracy_0.8584.h5')
print(model.evaluate(test_images, test_labels, verbose=False))
#%%
models_dir = 'models/'
model = create_model()
_ = model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
        epochs = 2, batch_size = 512,
        callbacks = [keras.callbacks.ModelCheckpoint(os.path.join(models_dir, 'epoch_{epoch:02d}_acc_{val_acc:.4f}.h5'),
        monitor='val_acc', save_weights_only=False, save_best_only=False)]
)
#%%
model = create_model()
print(model.evaluate(test_images, test_labels, verbose=False))
# %%
model = keras.models.load_model('models/epoch_02_acc_0.8440.h5')
# %%
model.summary()
print(model.evaluate(test_images, test_labels, verbose=False))
#%%
model.save_weights('tmp/manually_saved')
print(os.listdir('tmp'))
#%%
model.save('tmp/manually_saved_model.h5')
print(os.listdir('tmp'))
#%%
model.save('model_name')
print(os.listdir('model_name'))
#%%
model = keras.models.load_model('model_name')
print(model.evaluate(test_images, test_labels, verbose=False))