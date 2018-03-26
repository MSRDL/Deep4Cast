# Fix random seed for optimization
from keras.optimizers import RMSprop
import numpy as np

np.random.seed(42)

batch_size = 8
epochs = 50

loss = 'mse'
optimizer = RMSprop(lr=0.01)
metrics = ['mape']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

print('Training')
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs,
                    verbose=0, validation_split=0.0)