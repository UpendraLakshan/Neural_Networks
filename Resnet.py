from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
#from keras_resnet.models import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
#from tensorflow.keras.applications.resnet152 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.resnet152 import ResNet152
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model 
import matplotlib.pyplot as pyplot
import numpy as np 
import tensorflow


configuration = tensorflow.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session =tensorflow.compat.v1.Session(config=configuration)

img_height, img_width=(224,224)
batch_size=32

train_data_dir="Flowers_dataset/processed_data/train"
valid_data_dir="Flowers_dataset/processed_data/val"
test_data_dir="Flowers_dataset/processed_data/test"

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	validation_split=0.4)

train_generator=train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_height,img_width),
	batch_size=batch_size,
	class_mode='categorical',
	subset='training')

valid_generator=train_datagen.flow_from_directory(
	valid_data_dir,
	target_size=(img_height,img_width),
	batch_size=batch_size,
	class_mode='categorical',
	subset='validation')

test_generator=train_datagen.flow_from_directory(
	test_data_dir,
	target_size=(img_height,img_width),
	batch_size=1,
	class_mode='categorical',
	subset='validation')

x,y = test_generator.next()

base_model=ResNet50(include_top=False, weights='imagenet')
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
predictions=Dense(train_generator.num_classes,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
	layer.trainable=False
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_generator,
	epochs=5)

model.save('firstResNet50_2.h5')
test_loss,test_acc=model.evaluate(test_generator,verbose=2)
print('testAccuracy',test_acc)
