import AlexNet
from keras.callbacks import ModelCheckpoint, CSVLogger

model = AlexNet.built_model()
model.summary()

filepath = 'saved_models/weights-improvement-{epoch : 02d}-{val_acc : .2f}.h5'
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

log_csv = CSVLogger('my_logs.csv', separator=' , ', append=False)

callable_list=[checkpoint, log_csv]

'''
model.fit_generator(
 train_generator,
 steps_per_epoch=2000 // batch_size,
 epochs=500,
 validation_data=validation_generator,
 validation_steps=800 // batch_size,
 callbacks=callback_list   
)
'''
