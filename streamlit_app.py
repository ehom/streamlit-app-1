import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import platform
import time

def create_progress_cb(progress_bar):
    def inner_function(epoch, epochs):
        offseted_epoch = epoch + 1
        if offseted_epoch == epochs:
            progress_bar.progress(100, text="Epoch {} of {}: Training model...completed".format(offseted_epoch, epochs))
        else:
            progress_bar.progress(epoch/epochs, text="Epoch {} of {}: Training model...".format(offseted_epoch, epochs))
    return inner_function


class ourProgressCallback(tf.keras.callbacks.Callback):
  def __init__(self, progress_cb):
      self.progress_cb = progress_cb

  def on_epoch_end(self, epoch, logs={}):
      global max_loss

      epochs = self.params['epochs']
      loss = logs.get('loss')

      if max_loss is None:
          max_loss = loss

      self.progress_cb(epoch, epochs)
      loss_progress.progress(loss/max_loss, text="Loss: {}".format(round(loss, 4)))


def house_model(xs: np.array, ys: np.array):
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
    progress_cb = create_progress_cb(status_bar)
    model.fit(xs, ys, epochs=int(n_epochs), callbacks=[ourProgressCallback(progress_cb)])

    return model


if __name__ == "__main__":
    with st.sidebar:
        st.header('Intro to TensorFlow in AI, ML, and Deep Learning')
        st.write('Assignment 1')
        st.divider()
        st.write("**Application Properties**")
        st.markdown("* Python {}".format(platform.python_version()))
        st.markdown("* Tensorflow {}".format(tf.__version__))

    n_epochs = st.selectbox("How many epochs to train the pricing model?", ("1000", "500", "100"))

    global max_loss
    max_loss = None

    xs = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    ys = np.array([100, 150, 200, 250, 300, 350], dtype=float)

    # putting it in a dataframe, just so it can be displayed as a 2 column table
    df = pd.DataFrame({'xs': xs, 'ys': ys}, columns=['xs', 'ys'])

    st.write("House Pricing Model")
    df = df.set_index('xs')
    st.dataframe(df)

    if st.button('Train Model'):
        status_bar = st.progress(0, text="Training model...")
        loss_progress = st.progress(100, text="Tracking loss...")
        time.sleep(1)
        model = house_model(xs, ys)

        prediction = model.predict([7])
        st.write(f"The predicated price of that home is {prediction[0]}")
        st.balloons()

