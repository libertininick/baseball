import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import xgboost as xgb

n_examples = 10000
df = (pd.DataFrame({'x1': np.random.normal(0, 1, n_examples)
                   , 'x2': np.random.normal(-1, 2, n_examples)
                   , 'x3': np.random.normal(0, 1, n_examples)*np.random.normal(0, 1, n_examples)
                    , 'x4': np.random.normal(0, 1, n_examples)
                   }
                  )
      .eval('y = 3*x1*x2 - x3**2 + 0.5*x2 + x4')
      .drop(['x4'], axis='columns')
      )


ax = df['y'].plot(kind='hist', bins=30)
plt.show()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df.iloc[:, :-1]
                                                                            , df.iloc[:, -1]
                                                                            , test_size=0.3
                                                                            , random_state=1234)

dm_train = xgb.DMatrix(data=X_train, label=y_train)
dm_test = xgb.DMatrix(data=X_test, label=y_test)


def split(arr, sections):
    """
    Split an array into multiple sub-arrays.

    Args:
        arr (ndarray): Array to be divided into sub-arrays.
        sections (int): Number of sections to split array into. Leftovers are tacked onto last split.

    Returns:
        splits (list) A list of sub-arrays.
    """

    n = arr.shape[0]
    n_sections = n // sections
    n_trunc = n_sections*sections

    splits = np.split(arr[:n_trunc], sections)

    if n > n_trunc:
        splits[-1] = np.append(splits[-1], arr[n_trunc:])

    return splits


def loss_fxn(errors, sections=5):

    # Absolute value of errors vs. squared errors
    abs_errors = np.abs(errors)

    # Split errors into sections
    split_errors = split(abs_errors, sections)

    # Mean absolute error of each section
    split_mae = [np.mean(section) for section in split_errors]

    # Rank sections by mae
    split_ranks = np.array(split_mae).argsort().argsort() + 1

    # Apply rank scale to each absolute error based on its section
    split_scaled_ae = [section*rank for section, rank in zip(split_errors, split_ranks)]

    # Rescaling denom to normalize for ranking
    rescale_denom = np.sum([section.shape[0]*rank for section, rank in zip(split_errors, split_ranks)])/errors.shape[0]

    return np.concatenate(split_scaled_ae)/rescale_denom


def custom_obj_fxn(y_preds, y_true):
    errors = y_true - y_preds

    # Finite diff
    delta_x1 = 1e-5
    delta_x2 = 1e-1

    # Evaluate loss function
    f_x = loss_fxn(errors)
    f_x_dx1 = loss_fxn(errors + delta_x1)
    f_x_dx2 = loss_fxn(errors + delta_x2)
    f_x_dx1_dx2 = loss_fxn(errors + delta_x1 + delta_x2)

    grad = (f_x_dx1 - f_x)/delta_x1
    hess = ((f_x_dx1_dx2 - f_x_dx2)/delta_x1 - grad)/delta_x2

    return grad, hess


def custom_eval_fxn(y_preds, dm_y_true):
    y_true = dm_y_true.get_label()

    errors = y_true - y_preds

    return 'custom_mae', np.mean(loss_fxn(errors))


# Initialize XGBoost regression model
xgb_reg_model = xgb.XGBRegressor(max_depth=2
                                 , n_estimators=500
                                 , objective=custom_obj_fxn
                                 , seed=123
                                 )
# Fit XGBoost
xgb_reg_model.fit(X_train, y_train
                  , eval_set=[(X_train, y_train), (X_test, y_test)]
                  , eval_metric=custom_eval_fxn
                  , verbose=True)

df_progress = pd.concat([pd.DataFrame(xgb_reg_model.evals_result()['validation_0']).add_suffix('_train')
                            , pd.DataFrame(xgb_reg_model.evals_result()['validation_1']).add_suffix('_test')]
                        , axis='columns')

plt.style.use('seaborn-colorblind')
f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
(df_progress
 .filter(regex='^custom_mae')
 .plot(ax=ax1, kind='line')
 )

(df_progress
 .filter(regex='^rmse')
 .plot(ax=ax2, kind='line')
 )

plt.show()
