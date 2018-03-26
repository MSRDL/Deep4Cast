import matplotlib.pyplot as plt


plt.figure(figsize=(15,6))
plt.subplot(211)
plt.plot(history.epoch, history.history['loss'], 'r') 
# plt.plot(history.epoch, history.history['val_loss'], 'b')
plt.ylim(0, 2)
plt.subplot(212)
plt.plot(history.epoch, history.history['mean_absolute_percentage_error'], 'r')
# plt.plot(history.epoch, history.history['val_mean_absolute_percentage_error'], 'b')
plt.ylim(0, 200)

print('Min. training loss: {0:.3f}'.format(min(history.history['loss'])))
# print('Min. validation loss: {1:.3f}'.format(min(history.history['val_loss'])))

print('Testing')
test(model)