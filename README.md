## air-writing-recognition
Air-Writing Recognition using Deep Convolutional and Recurrent Neural Network Architectures - International Conference on Frontiers of Handwriting Recognition (ICFHR)

## Experiment 1


## Experiment 2

# cross-validation

# train distinct models

```
cd exp2
python air_test.py --model {cnn, cnn-lstm, lstm, tcn_dynamic}
```

# train fuzzy models

```
cd exp2\fuzzy_LSTMWithCNN
python air_test.py
```

# fine-tuning CNN and LSTM on Participants #5 and #8
```
python air_test.py --model cnn --run_all_folds False --Nf 5 --n_injections 2 --epochs 20
python air_test.py --model cnn --run_all_folds False --Nf 5 --n_injections 5 --epochs 20
python air_test.py --model cnn --run_all_folds False --Nf 8 --n_injections 2 --epochs 20
python air_test.py --model cnn --run_all_folds False --Nf 8 --n_injections 5 --epochs 20

python air_test.py --model lstm --run_all_folds False --Nf 5 --n_injections 2 --epochs 200
python air_test.py --model lstm --run_all_folds False --Nf 5 --n_injections 5 --epochs 200
python air_test.py --model lstm --run_all_folds False --Nf 8 --n_injections 2 --epochs 200
python air_test.py --model lstm --run_all_folds False --Nf 8 --n_injections 5 --epochs 200
```
