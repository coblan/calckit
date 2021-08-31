import matplotlib.pyplot as plt

def draw_history(history,start=None,end=None):
    history_dict = history.history
    if start:
        loss_values = history_dict['loss'][start:end]
        val_loss_values = history_dict['val_loss'][start:end]
    else:
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()