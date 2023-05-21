# Notes:

- optimizer: Adam
- Loss: categorical_crossentropy
- metrics: accuracy
- epocks: 50
- callback: earlystopping after 3 epochs if the val_loss didn't decrease
- train, test, validation:  42k, 10k, 8k
- Batch size: Not Specified (we need to use the same batch for all architectures)
- No preprocessing (can be modified and do some of them but before must be satetd here and on the group)
