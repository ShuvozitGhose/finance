import matplotlib.pyplot as plt
import numpy as np

def plot_close_volume(df):
    # Plot daily IBM closing prices and volume
    fig = plt.figure(figsize=(15,10))
    st = fig.suptitle("S&P 500 Close Price and Volume", fontsize=20)
    st.set_y(0.92)

    ax1 = fig.add_subplot(211)
    ax1.plot(df['Close'], label='S&P 500 Close Price')
    ax1.set_xticks(range(0, df.shape[0], 1464))
    ax1.set_xticklabels(df['Date'].loc[::1464])
    ax1.set_ylabel('Close Price', fontsize=18)
    ax1.legend(loc="upper left", fontsize=12)

    ax2 = fig.add_subplot(212)
    ax2.plot(df['Volume'], label='S&P 500 Volume')
    ax2.set_xticks(range(0, df.shape[0], 1464))
    ax2.set_xticklabels(df['Date'].loc[::1464])
    ax2.set_ylabel('Volume', fontsize=18)
    ax2.legend(loc="upper left", fontsize=12)

    # plt.show()
    plt.savefig('plot_close_volume.png')


def display_results(seq_len, train_data, val_data, test_data, train_pred, val_pred, test_pred):
    '''Display results'''

    fig = plt.figure(figsize=(15,20))
    st = fig.suptitle("Transformer + TimeEmbedding Model", fontsize=22)
    st.set_y(0.92)

    #Plot training data results
    ax11 = fig.add_subplot(311)
    ax11.plot(train_data[:, 3], label='S&P 500 Closing Returns')
    ax11.plot(np.arange(seq_len, train_pred.shape[0]+seq_len), train_pred, linewidth=3, label='Predicted S&P 500 Closing Returns')
    ax11.set_title("Training Data", fontsize=18)
    ax11.set_xlabel('Date')
    ax11.set_ylabel('S&P 500 Closing Returns')
    ax11.legend(loc="best", fontsize=12)

    #Plot validation data results
    ax21 = fig.add_subplot(312)
    ax21.plot(val_data[:, 3], label='S&P 500 Closing Returns')
    ax21.plot(np.arange(seq_len, val_pred.shape[0]+seq_len), val_pred, linewidth=3, label='Predicted S&P 500 Closing Returns')
    ax21.set_title("Validation Data", fontsize=18)
    ax21.set_xlabel('Date')
    ax21.set_ylabel('S&P 500 Closing Returns')
    ax21.legend(loc="best", fontsize=12)

    #Plot test data results
    ax31 = fig.add_subplot(313)
    ax31.plot(test_data[:, 3], label='S&P 500 Closing Returns')
    ax31.plot(np.arange(seq_len, test_pred.shape[0]+seq_len), test_pred, linewidth=3, label='Predicted S&P 500 Closing Returns')
    ax31.set_title("Test Data", fontsize=18)
    ax31.set_xlabel('Date')
    ax31.set_ylabel('S&P 500 Closing Returns')
    ax31.legend(loc="best", fontsize=12)

    #plt.show()  
    plt.savefig('display_results.png')


def display_metrics(history):
    # Model metrics
    '''Display model metrics'''

    fig = plt.figure(figsize=(15,20))
    st = fig.suptitle("Transformer + TimeEmbedding Model Metrics", fontsize=22)
    st.set_y(0.92)

    #Plot model loss
    ax1 = fig.add_subplot(311)
    ax1.plot(history.history['loss'], label='Training loss (MSE)')
    ax1.plot(history.history['val_loss'], label='Validation loss (MSE)')
    ax1.set_title("Model loss", fontsize=18)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend(loc="best", fontsize=12)

    #Plot MAE
    ax2 = fig.add_subplot(312)
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title("Model metric - Mean average error (MAE)", fontsize=18)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean average error (MAE)')
    ax2.legend(loc="best", fontsize=12)

    #Plot MAPE
    ax3 = fig.add_subplot(313)
    ax3.plot(history.history['mape'], label='Training MAPE')
    ax3.plot(history.history['val_mape'], label='Validation MAPE')
    ax3.set_title("Model metric - Mean average percentage error (MAPE)", fontsize=18)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean average percentage error (MAPE)')
    ax3.legend(loc="best", fontsize=12)

    #plt.show()
    plt.savefig('display_metrics.png')