python air_test.py --model {cnn, cnn-lstm, lstm, tcn_dynamic}

python air_test.py --model cnn --model_dir ../extra_models/ --run_all_folds False --n_injections 1 --epochs 100


<!-- python air_test.py --model cnn --run_all_folds False --Nf 8 --n_injections 2 --epochs 10 -->

python air_test.py --model cnn --run_all_folds False --Nf 5 --n_injections 2 --epochs 20
python air_test.py --model cnn --run_all_folds False --Nf 5 --n_injections 5 --epochs 20
python air_test.py --model cnn --run_all_folds False --Nf 8 --n_injections 2 --epochs 20
python air_test.py --model cnn --run_all_folds False --Nf 8 --n_injections 5 --epochs 20


python air_test.py --model lstm --run_all_folds False --Nf 5 --n_injections 2 --epochs 200
python air_test.py --model lstm --run_all_folds False --Nf 5 --n_injections 5 --epochs 200
python air_test.py --model lstm --run_all_folds False --Nf 8 --n_injections 2 --epochs 200
python air_test.py --model lstm --run_all_folds False --Nf 8 --n_injections 5 --epochs 200
<!-- python air_test.py --model lstm --run_all_folds False --Nf 1 --n_injections 2 --epochs 200
python air_test.py --model lstm --run_all_folds False --Nf 1 --n_injections 5 --epochs 200 -->