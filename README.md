# Self Adaptive LSTM for real time predictions

## Usage

### Setup project dependencies

```{sh}
pip install .
```

### Start hyperparameter search procedure

The following command will generate 30 random hyperparameter samples to create a baseline dataset used to fit the first Bayesian surrogate model (Space Filling Procedure). It thereafter generates "smart" hyperparameter samples that are likely to minimize the loss function of the model over the test dataset. The total amount of "smart" samples is equal to sampleBudget - spaceFilling = 20.

The total times the underlying model is fit is controlled by the sampleBudget parameter.

sampleBudget must be greater than or equal to spaceFilling.

```{sh}
python main -spaceFilling=30 -sampleBudget=50
```
