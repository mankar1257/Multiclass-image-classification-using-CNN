 
"""
@author: Vaibhav R Mankar
"""

'''
    MultiClass Prediction Based on Convulutional Nural Network 
      


'''

#---------------------------------------------Importing Liberies------------------------------
 
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator





#-----------------------------------------------Data-Loading------------------------------------

 
samples_per_epoch = 1000
validation_steps = 300

def load_dataset():
    
    train_data_path = '/home/vaibhav/Documents/dataset2/seg_train/seg_train'
    validation_data_path = '/home/vaibhav/Documents/dataset2/seg_test/seg_test'
	# load dataset
    batch_size = 32
    img_width, img_height = 150, 150
    
    
    train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
	
    return train_generator ,validation_generator 




#---------------------------------------------CNN-Artitecture -------------------------------------------
    



def CNNmodel():
    
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2))) 
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(6, activation='softmax'))
	# compile model
	opt = SGD(lr=0.002, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 

#-----------------------------------------------------------ploting-Learning-curves-------------------

def Plot_Results(history):
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    plt.plot(loss_train, 'g', label='Training loss')
    plt.plot(loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('plot1.png')
    
    loss_train = history.history['acc']
    loss_val = history.history['val_acc']
    plt.plot(loss_train, 'g', label='Training accuracy')
    plt.plot(loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('plot1.png')
   


#------------------------------------------------------------Running-the-Model----------------------------
    
    
    
def Model():
	# load dataset
	train_generator , validation_generator = load_dataset()
    
	# define model
	model = CNNmodel()
    
	# fit model
	history = model.fit_generator( 
                                    train_generator,
                                    samples_per_epoch=samples_per_epoch,
                                    epochs=10,
                                    validation_data=validation_generator,
                                    validation_steps=validation_steps)
    
	# learning curves
	Plot_Results(history)
 

Model()