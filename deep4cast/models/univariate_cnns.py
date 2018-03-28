# from keras.layers.advanced_activations import ELU
# # Hyper parameters
# n_layers = 1
# activation = ELU()
# batch_size = 8
# epochs = 50

# print('Creating Model')
# sequence_input = Input(shape=(sequence_length, dim))
# x = Conv1D(64, 5, activation=activation, init='glorot_normal')(sequence_input)
# # dim: 16 x 64
# x = MaxPooling1D(3, stride=1)(x)
# # x = AveragePooling1D(pool_length=2, stride=1)(x)
# # dim: 14 x 64
# x = Conv1D(64, 3, activation=activation, init='glorot_normal')(x)
# # dim: 12 x 64
# x = MaxPooling1D(4, stride=2)(x)  # global max pooling
# # x = AveragePooling1D(pool_length=4, stride=1)(x)
# # dim: 5 x 64
# x = Conv1D(128, 3, activation=activation, init='glorot_normal')(x)
# # dim: 3 x 128
# x = MaxPooling1D(3, stride=1)(x)  # global max pooling
# # x = AveragePooling1D(pool_length=4, stride=1)(x)
# # dim: 1 x 128
# x = Flatten()(x)
# x = Dense(128, activation=activation, init='glorot_normal')(x)
# # x = Dropout(0.5)(x)
# preds = Dense(dim)(x)
# model = Model(sequence_input, preds)

# # sgd = Nadam(lr=999, schedule_decay=0.1)
# # sgd = Nadam(lr=0.1, schedule_decay=0.01)
# # sgd = RMSprop(lr=0.1)
# # sgd = SGD(lr=0.01, momentum=0.9, decay=0.1, nesterov=False)
# # model.compile(loss='mape', optimizer=sgd, metrics=['mape'])
# sgd = SGD(lr=0.1, momentum=0.9, decay=0.1, nesterov=False)
# model.compile(loss='mse', optimizer=sgd, metrics=['mape'])

# print('Training')
# history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, 
#                     verbose=0, validation_split=0.0)

# plt.figure(figsize=(15,6))
# plt.subplot(211)
# plt.plot(history.epoch, history.history['loss'], 'r') 
# # plt.plot(history.epoch, history.history['val_loss'], 'b')
# plt.ylim(0, 2)
# plt.subplot(212)
# plt.plot(history.epoch, history.history['mean_absolute_percentage_error'], 'r')
# #plt.plot(history.epoch, history.history['val_mean_absolute_percentage_error'], 'b')
# plt.ylim(0, 200)

# print('Min. training loss: {0:.3f}'.format(min(history.history['loss'])))
# # print('Min. validation loss: {1:.3f}'.format(min(history.history['val_loss'])))
# print('Testing')
# test(model)