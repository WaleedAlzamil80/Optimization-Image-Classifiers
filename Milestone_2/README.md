# Notes:

- optimizer: Adam
- Loss: categorical_crossentropy
- metrics: accuracy
- epocks: 50
- callback: earlystopping after 3 epochs if the val_loss didn't decrease
- train, test, validation:  42k, 10k, 8k
- batch_size: 256
- No preprocessing (can be modified and do some of them but before must be satetd here and on the group)
- Use the following function to save the progress in case the colabe is disconnected
```
def download_history(history, name, predictions, test):

  with open('history_' + name + '.pkl', 'wb') as f:
    pickle.dump(history.history, f)

  files.download('history_' + name + '.pkl')

  np.save("y_pred_"+ name + ".npy", predictions)

  files.download("y_pred_"+ name + ".npy")

  np.save("y_test_"+ name + ".npy", test)

  files.download("y_test_"+ name + ".npy")
```
