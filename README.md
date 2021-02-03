# `Pytorch chatbot`
A simple chatbot implemented using `pytorch` library.
It uses `Seq2Seq` model built with multi-layered Gated Recurrent Unit 
encoder and decoder.  
Model is configured for 
[Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).  
By modifying `Loader` class and `config.yaml` it could be used on other datasets as well.  
Repository is inspired by a [Pytorch Chatbot Tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html).

### Setup  
Project originally used `Python 3.8`.  
To install required libraries execute:
```
pip install -r requirments.txt
```
-----
### Training
Training script will download and preprocess data needed for training the model, results will be saved
to a directory specified in a `config.yaml` file.  
To train model simply run:

```
python train.py
```
---
### Chatting
After model is trained chat with a model can be performed by running.
```
python chat.py
```