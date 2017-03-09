from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.visualize_util import plot
import numpy

dataset = numpy.loadtxt("random_dataset.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:5]
Y = dataset[:,5]
Y_reg = dataset[:,6]

model = Sequential()
model.add(Dense(1, input_dim=5, init='uniform', activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y_reg, nb_epoch=500, batch_size=1)

scores = model.evaluate(X, Y_reg)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


test = X[0:10,:]
predictions = model.predict(test)
print(predictions)

plot(model, to_file='model.png', show_shapes='true')