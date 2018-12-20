import tensorflow as tf
import numpy as np

class ReversiAgent:
    def __init__(self, Q=None):
        if Q is None:
            self._Q = self.__createmodel__()
        else:
            self._Q = tf.keras.models.clone_model(Q)
            self._Q.set_weights(Q.get_weights())

        self._targetQ =  tf.keras.models.clone_model(self._Q)
        self._targetQ.set_weights(self._Q.get_weights())


    def get_next_move(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(64)
        else:
            return np.argmax(self.get_move_values(state))


    def get_move_values(self, state):
        return self._Q.predict(self.__processtate__(state)).reshape((64,))


    def update_Q(self, memory_batch):
        states = np.vstack([mem[0] for mem in memory_batch])
        targets = np.vstack([self.__computetargetvalues__(mem) for mem in memory_batch])

        self._Q.fit(self.__processtate__(states), targets, batch_size=len(memory_batch), epochs=1, verbose=0)


    def update_targetQ(self):
        self._targetQ.set_weights(self._Q.get_weights())


    def clone(self):
        return ReversiAgent(self._Q)


    def save(self, filename):
        tf.keras.models.save_model(self._Q, filename)


    def __processtate__(self, state):
        return state.reshape((-1, 8, 8, 1))


    def __computetargetvalues__(self, transition):
        state = transition[0]
        action = transition[1]
        reward = transition[2]
        next_state = transition[3]
        terminal = transition[4]

        targets =  self._targetQ.predict(self.__processtate__(state)).reshape((64,))
        target_value = reward if terminal else reward + np.max(self.get_move_values(next_state))
        targets[action] = target_value

        return targets


    def __createmodel__(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16, (2,2), input_shape=(8,8,1)))
        model.add(tf.keras.layers.Conv2D(8, (4,4)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(64))

        model.compile(optimizer='adam', loss='mse')

        return model

