
# In[53]:


from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt


# In[54]:


image_width, image_height = 150,150
mask_dir = os.path.join('C:\\Users\\tanmo\\Desktop\\Tanmoy Files\\Projects\\Spectrum Analysis Project [PAPER]\\spectogram\\data\\Train\\negative')
validation_dir = os.path.join('C:\\Users\\tanmo\\Desktop\\Tanmoy Files\\Projects\\Spectrum Analysis Project [PAPER]\\spectogram\\data\\Test\\negative')

print('Negative')
print (len(os.listdir(mask_dir)))
print (len(os.listdir(validation_dir)))


# In[55]:


val_dir = 'C:\\Users\\tanmo\\Desktop\\Tanmoy Files\\Projects\\Spectrum Analysis Project [PAPER]\\spectogram\\data\\Test'
train_dir = 'C:\\Users\\tanmo\\Desktop\\Tanmoy Files\\Projects\\Spectrum Analysis Project [PAPER]\\spectogram\\data\\Train'

nb_train_samples = 80
nb_validation_samples = 40
epochs = 50
batch_size = 8


training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    )

validation_datagen = ImageDataGenerator(rescale= 1./255)


# In[56]:


training_generator = training_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    class_mode='categorical',
    batch_size= 30
    )

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(150,150),
    class_mode = 'categorical',
    batch_size = 30
    )


# In[67]:


model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()


# In[68]:


model.compile(loss='binary_crossentropy',
             optimizer = 'rmsprop',
             metrics=['accuracy'])


# In[76]:


history = model.fit(training_generator,
                   steps_per_epoch=nb_train_samples // batch_size,
                   epochs = epochs,
                   validation_data=validation_generator,
                   validation_steps=nb_validation_samples // batch_size)

# In[77]:


model.save('new.h5')


# In[ ]:

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


