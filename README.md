Based on the paper Siamese Recurrent Architectures for Learning Sentence Similarity by Jonas Mueller and Aditya Thyagarajan

Download pretrained word2vec model from [here](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)

Download Quera Dataset [here](https://www.kaggle.com/c/quora-question-pairs/data)

Trained models with different configurations can be downloaded [here](https://drive.google.com/open?id=1suscHM-xF7w4KFwvZLUEsHEu8tUSo8-f)

the data and word2vec model should be put into data directory and the pretrained models should be put into output directory

install requirement by
```
pip3 install -r requirement.txt
```
run the line below to train
```
python3 training.py
```
run the line below to test on validate set
```
python3 evaluate.py
```
modify hyperparameters.py to change hyperparameters and select pretrained model

