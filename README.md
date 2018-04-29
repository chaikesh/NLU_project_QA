### Question Answering 

# Setup Environment 

1. Make sure you have Anaconda installed in your PC and Environment is set for Anaconda and make requirement.sh executable. Now run in terminal 


```diff
cd [Folder where you have downloaded git repositry]
```

```diff
chmod +x get_started.sh
```

```diff
./get_started.sh
```

Above command will create the new conda environment "squad". and download the squad datset and GLOVE word2vec embedding matrix data. Also, It will download tensorflow=1.4.1 and other useful packages in this environemnt and data folder

# Run Train
```diff
source activate squad
```

```diff
cd [extracted_Folder]/code/
```

```diff
python main.py --experiment_name=qa --mode=train
```

above command will start training the model and it will create a folder "experiment" which has model checkpoints saved

# Seeing Example results when we trained some iterations
```diff
source activate squad
```
```diff
python main.py --experiment_name=qa --mode=show_examples
```

# Visualizing Loss and EM/F1 score during training or finished training
```diff
tensorboard --logdir=. --port=5678
```
