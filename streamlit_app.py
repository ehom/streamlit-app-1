import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

st.header('Introduction to TensorFlow in Artifical Intelligence, Machine Learning, and Deep Learning')

st.write('Assignment 1')

using_tf_version = "Using TensorFlow {}".format(tf.__version__)

st.write(using_tf_version)

st.sidebar.write("testing")

n_epochs = st.selectbox("How many epochs to train the pricing model?", ("1000", "500", "100"))

class ourProgressCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      loss = logs.get('loss')
      if epoch == n_epochs:
          st.write(f"epoch: {epoch}, loss: {loss}")

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
    st.write('Training Model..')
    # Get your trained model

    xs = np.array([1, 2, 3, 4, 5, 6], dtype=int)
    ys = np.array([100, 150, 200, 250, 300, 350], dtype=float)

    # putting it in a dataframe, just so it can be displayed as a 2 column table
    df = pd.DataFrame({'xs': xs, 'ys': ys}, columns=['xs', 'ys'])

    df = df.set_index('xs')
    st.dataframe(df)

    with st.spinner(text="training..."):
        model = house_model(xs, ys)
        st.success("training...completed")

    prediction = model.predict([7])
    st.write(f"The predicated price of that home is {prediction[0]}")
    st.balloons()


