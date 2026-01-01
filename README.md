The implementation is based on [keras-rl](https://github.com/keras-rl/keras-rl) and [OpenAI baselines](https://github.com/openai/baselines) frameworks.

- `gym-control`: Classic control games

## Dependencies
- python 3.5
- tensorflow 1.10.0, keras 2.1.0
- gym, scipy, scipy, joblib, keras
- progressbar2, mpi4py, cloudpickle, opencv-python, h5py, pandas

Note: make sure that you have successfully installed the baseline package and other packages following (using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) to create virtual environment):
```

pip install -r requirements.txt
pip install -e .
```

### Training
```
sh scripts/train-qlearn.sh
sh scripts/train-dqn.sh
```
### Visualizing
```
sh scripts/visualize.sh
```

## Reproduce the Results
To reproduce all the results, please refer to `scripts/` folders:
- `gym-control/scripts`
- `train-qlearn.sh` (Q-Learning)
- `train-dqn.sh` (DQN)

	
The logs and models will be saved automatically. We provide `results_single.py` for getting the averaged scores:
```
python -m baselines.results_single --log_dir logs-alien
```






