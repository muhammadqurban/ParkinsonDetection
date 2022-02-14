from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

img_width, img_height = 200, 200
train_data_dir = './dataset'
batch_size = 16
epochs = 20


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1. / 255,
                             rotation_range=1,
                             brightness_range=[1.0,1.1],
                             fill_mode='nearest',
                             validation_split=0.2)

train_generator = train_datagen.flow_from_directory(train_data_dir,batch_size = batch_size, class_mode ='categorical', target_size=(img_width,img_height), subset='training')
validation_generator = train_datagen.flow_from_directory(train_data_dir,batch_size = batch_size, class_mode ='categorical', target_size=(img_width,img_height), subset='validation')
checkpoint = ModelCheckpoint('model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto',save_weights_only=True)
history = model.fit_generator(train_generator, steps_per_epoch = train_generator.samples // batch_size, epochs = epochs, validation_data = validation_generator, validation_steps = validation_generator.samples // batch_size,callbacks=[checkpoint])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")