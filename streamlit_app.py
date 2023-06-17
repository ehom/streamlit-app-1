import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import platform
import time

sidebar = st.sidebar

with sidebar:
    st.header('Intro to TensorFlow in AI, ML, and Deep Learning')
    st.write('Assignment 1')
    st.divider()
    st.write("**Application Properties**")
    st.markdown("* Python {}".format(platform.python_version()))
    st.markdown("* Tensorflow {}".format(tf.__version__))

xs = np.array([1, 2, 3, 4, 5, 6], dtype=int)
ys = np.array([100, 150, 200, 250, 300, 350], dtype=float)

# putting it in a dataframe, just so it can be displayed as a 2 column table
df = pd.DataFrame({'xs': xs, 'ys': ys}, columns=['xs', 'ys'])

st.write("House Pricing Model")
df = df.set_index('xs')
st.dataframe(df)


n_epochs = st.selectbox("How many epochs to train the pricing model?", ("1000", "500", "100"))


class ourProgressCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      epoches = self.params['epochs']
      if epoch == epoches - 1:
          loss = logs.get('loss')
          st.write(f"In the last epoch ({epoch}), loss was {round(loss, 4)}")
          status_bar.progress(100, text="Training model...completed")
      else:
          status_bar.progress(epoch/epoches, text="Training model...")

def house_model(xs, ys):
    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    # Declare model inputs and outputs for training

    # Define your model (should be a model with 1 dense layer and 1 unit)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function

    model.compile(optimizer="sgd", loss="mean_squared_error")

    # Train your model for 1000 epochs by feeding the i/o tensors

    model.fit(xs, ys, epochs=int(n_epochs), callbacks=[ourProgressCallback()])

    return model


if st.button('Train Model'):
    status_bar = st.progress(0, text="Training model...")
    time.sleep(1)
    model = house_model(xs, ys)

    prediction = model.predict([7])
    st.write(f"The predicated price of that home is {prediction[0]}")
    st.balloons()

