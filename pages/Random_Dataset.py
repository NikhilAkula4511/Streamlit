import streamlit as st
st.set_page_config(
    page_title="Streamlit Application For Perceptronic Model",
    page_icon="👋",
)


st.sidebar.success("Select a page above.")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import keras
st.title("Tensorflow_Nikhil_Elite13")
st.header("Random Data application")
from sklearn.datasets import make_classification,make_regression
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import InputLayer,Dense




#num_samples
num_samples = st.sidebar.slider("No_of samples", min_value=1, max_value=100000, value=10, step=1)

#random state or not

random_sates = st.sidebar.slider("Select Random state" ,min_value=1, max_value=100, value=0, step=1)

# Learning rate selection
learning_rates =  st.sidebar.slider("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001,
                                    help="Adjust the learning rate for the model.")

sgd=SGD(learning_rate=learning_rates)

hidden_layers = st.sidebar.number_input("Hidden Layers", min_value=1, step=1)
hidden_layers_config = []
for layer in range(1, hidden_layers + 1):
    neurons = st.sidebar.number_input(f"Number of Neurons in Layer {layer}", min_value=1, step=1)
    activation = st.sidebar.selectbox(f"Activation Function for Layer {layer}", ["relu", "sigmoid", "tanh","softmax"])
    hidden_layers_config.append((neurons, activation))

model = Sequential()
model.add(InputLayer(input_shape=(2,)))
for neurons, activation in hidden_layers_config:
    model.add(Dense(neurons, activation=activation, use_bias=True))
num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, step=1)
batch_size = st.sidebar.number_input("Batch Size", min_value=1, step=1)

if st.sidebar.button("Submit", type="primary"):


    fv,cv=make_classification(n_samples=num_samples,n_features=2,n_informative=2,
                        n_redundant=0,n_repeated=0,n_classes=2,n_clusters_per_class=1, class_sep=3,random_state=random_sates)
    df = pd.DataFrame(fv, columns=["Feature 1", "Feature 2"])
    df["Class Label"] = cv

    


    # Create scatter plot
    st.write("### Correlation between features wrt Class Label")

    fig = sns.scatterplot(data=df, x="Feature 1", y="Feature 2", hue="Class Label", palette="viridis")
    fig.set_xlabel('Feature 1')
    fig.set_ylabel('Feature 2')
    st.pyplot(fig.figure)
    y=df['Class Label']
    X=df[['Feature 1','Feature 2']]
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_tset = train_test_split(fv,cv,test_size=0.3,stratify=cv)

    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    std=StandardScaler()
    X_train=std.fit_transform(X_train)
    X_test=std.transform(X_test)



    model.summary()
    model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

    history=model.fit(X_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_split=0.3,steps_per_epoch=700//20)
    history
    # Streamlit app
    st.title("Training and Validation Loss Plot")
    st.write("### Plotting Loss Curves")

    # Create plot
    fig,ax = plt.subplots()
    plt.plot(range(1, num_epochs+1), history.history['loss'], label='Train')
    plt.plot(range(1, num_epochs+1), history.history['val_loss'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    st.pyplot(fig)

    from mlxtend.plotting import plot_decision_regions

    # Plot decision regions
    fig, ax = plt.subplots()
    plot_decision_regions(X_test, y_tset, clf=model, ax=ax)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Regions')
    st.pyplot(fig)