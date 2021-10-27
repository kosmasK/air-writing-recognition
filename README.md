# air-writing-recognition
This is the code for the experiments presented in the article [Air-Writing Recognition using Deep Convolutional and Recurrent Neural Network Architectures - International Conference on Frontiers of Handwriting Recognition (ICFHR 2020)](https://ieeexplore.ieee.org/abstract/document/9257775).

# Experiment 1

```
cd exp1/cnn_static_writing
python air_test.py {--train False}

cd exp1/lstm_air_writing
python air_test.py {--use_cnn False} {--bidirectional True} {--train False}

cd exp1/tcn_dynamic
python air_test.py {--train False}

cd exp1/tcn_static
python air_test.py {--train False}

cd exp1/fuzzy_LSTMWithCNN
python air_test.py {--train False}
```


# Experiment 2 (participant cross-validation)


## train distinct models

```
cd exp2
python air_test.py --model {cnn, cnn-lstm, lstm, tcn_dynamic}
```

## train fuzzy models

```
cd exp2/fuzzy_LSTMWithCNN
python air_test.py
```

## fine-tuning CNN and LSTM on Participants #5 and #8
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

## Demo
https://apps.ilsp.gr:5006/airwriting/

You must plug in a Leap Motion Controller and have the proper drivers installed in order to use the air-writing environment. Place the sensor on the table, in front of and parallel to your computer monitor. Move your index finger right above the sensor and start drawing digits in free space.

<p align="center">
    <img src="airwr_vid2c.gif" width="400">
</p>
