import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


FEATURE_FOLDER = 'clean_data'    
    
RANDOM_STATE = 0

RESULT_FILE = "4_prediction_detailed_result.txt"

if RESULT_FILE in os.listdir():
    os.remove(RESULT_FILE)

def print_to_file(message):
    with open(RESULT_FILE, "a+") as f:
        f.write(message + "\n")

for num_future_days in [1, 7, 30, 60]:
    print_to_file("-------------------------------------")
    print_to_file(f"Result for future {num_future_days} day(s):")
    clean_data = pd.read_csv(f"{FEATURE_FOLDER}/ALL_TICKERS_{num_future_days}_DAYS.csv")

    # Prepare features and test dataframe
    train_clean_data, test_clean_data = train_test_split(
        clean_data, 
        test_size=0.1, 
        shuffle=True, 
        random_state=RANDOM_STATE
    )
    train_features = train_clean_data[[
        'num_environment_sentences', 'environment_sentiment_score',
        'num_social_sentences', 'social_sentiment_score',
        'num_economy_sentences', 'economy_sentiment_score',
        'hist_1_day_return', 'hist_7_day_return',
        'hist_30_day_return', 'hist_60_day_return'
    ]]
    train_target = train_clean_data[['return']]
    test_features = test_clean_data[[
        'num_environment_sentences', 'environment_sentiment_score',
        'num_social_sentences', 'social_sentiment_score',
        'num_economy_sentences', 'economy_sentiment_score',
        'hist_1_day_return', 'hist_7_day_return',
        'hist_30_day_return', 'hist_60_day_return'
    ]]
    test_target = test_clean_data[['return']]
    
    # Calculate RMSE for Lazy Predictor
    lazy_predictor_train_rmse = mean_squared_error(
        [train_target.mean()] * len(train_target),
        train_target
    )
    lazy_predictor_test_rmse = mean_squared_error(
        [test_target.mean()] * len(test_target),
        test_target
    )
    print_to_file(f'    - Lazy Predictor - Train MSE: {lazy_predictor_train_rmse}')
    print_to_file(f'    - Lazy Predictor - Test MSE: {lazy_predictor_test_rmse}')

    # config_1 result
    config_1_feature_columns = [
        'num_environment_sentences', 'environment_sentiment_score',
        'num_social_sentences', 'social_sentiment_score',
        'num_economy_sentences', 'economy_sentiment_score'
    ]
    config_1_search_space = {
        "n_estimators": [50, 100, 200, 500],
        "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15, 18, 20],
        "max_depth": [None, 2, 3, 4],
        "max_features": [2, 4, 6]
    }
    config_1_model_base_rf = RandomForestRegressor(random_state=RANDOM_STATE)
    config_1_gs = GridSearchCV(
        estimator=config_1_model_base_rf, 
        param_grid=config_1_search_space,
        scoring='neg_root_mean_squared_error',
        refit=True,
        cv=5,
        verbose=3
    )
    config_1_gs.fit(train_features[config_1_feature_columns], train_target.to_numpy().reshape(-1,))
    config_1_model = config_1_gs.best_estimator_
    print_to_file(f"Best model for config 1: {str(config_1_model)}")
    config_1_train_rmse = mean_squared_error(
        config_1_model.predict(train_features[config_1_feature_columns]),
        train_target
    )
    config_1_test_rmse = mean_squared_error(
        config_1_model.predict(test_features[config_1_feature_columns]),
        test_target
    )
    
    print_to_file(f'    - Config 1 - Train MSE: {config_1_train_rmse}')
    print_to_file(f'    - Config 1 - Test MSE: {config_1_test_rmse}')
    print_to_file("     - Config 1 - Feature Importance: ")
    print_to_file(str(pd.Series(config_1_model.feature_importances_, index = config_1_feature_columns)))

    # config_2 result
    config_2_feature_columns = ['hist_1_day_return', 'hist_7_day_return', 'hist_30_day_return', 'hist_60_day_return']
    config_2_search_space = {
        "n_estimators": [50, 100, 200, 500],
        "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15, 18, 20],
        "max_depth": [None, 2, 3, 4],
        "max_features": [2, 4]
    }
    config_2_model_base_rf = RandomForestRegressor(random_state=RANDOM_STATE)
    config_2_gs = GridSearchCV(
        estimator=config_2_model_base_rf, 
        param_grid=config_2_search_space,
        scoring='neg_root_mean_squared_error',
        refit=True,
        cv=5,
        verbose=3
    )
    config_2_gs.fit(train_features[config_2_feature_columns], train_target.to_numpy().reshape(-1,))
    config_2_model = config_2_gs.best_estimator_
    print_to_file(f"Best model for config 2: {str(config_2_model)}")
    config_2_train_rmse = mean_squared_error(
        config_2_model.predict(train_features[config_2_feature_columns]),
        train_target
    )
    config_2_test_rmse = mean_squared_error(
        config_2_model.predict(test_features[config_2_feature_columns]),
        test_target
    )
    
    print_to_file(f'    - Config 2 - Train MSE: {config_2_train_rmse}')
    print_to_file(f'    - Config 2 - Test MSE: {config_2_test_rmse}')
    print_to_file("     - Config 2 - Feature Importance: ")
    print_to_file(str(pd.Series(config_2_model.feature_importances_, index = config_2_feature_columns)))

    # config_3 result
    config_3_feature_columns = ['num_environment_sentences', 'environment_sentiment_score',
                                'num_social_sentences', 'social_sentiment_score',
                                'num_economy_sentences', 'economy_sentiment_score',
                                'hist_1_day_return', 'hist_7_day_return', 'hist_30_day_return', 'hist_60_day_return'
    ]
    config_3_search_space = {
        "n_estimators": [50, 100, 200, 500],
        "min_samples_leaf": [1, 2, 3, 5, 7, 10, 15, 18, 20],
        "max_depth": [None, 2, 3, 4],
        "max_features": [2, 4, 6, 8, 10]
    }
    config_3_model_base_rf = RandomForestRegressor(random_state=RANDOM_STATE)
    config_3_gs = GridSearchCV(
        estimator=config_3_model_base_rf, 
        param_grid=config_3_search_space,
        scoring='neg_root_mean_squared_error',
        refit=True,
        cv=5,
        verbose=3
    )
    config_3_gs.fit(train_features[config_3_feature_columns], train_target.to_numpy().reshape(-1,))
    config_3_model = config_3_gs.best_estimator_
    print_to_file(f"Best model for config 3: {str(config_3_model)}")
    config_3_train_rmse = mean_squared_error(
        config_3_model.predict(train_features[config_3_feature_columns]),
        train_target
    )
    config_3_test_rmse = mean_squared_error(
        config_3_model.predict(test_features[config_3_feature_columns]),
        test_target
    )
    
    print_to_file(f'    - Config 3 - Train MSE: {config_3_train_rmse}')
    print_to_file(f'    - Config 3 - Test MSE: {config_3_test_rmse}')
    print_to_file("     - Config 3 - Feature Importance: ")
    print_to_file(str(pd.Series(config_3_model.feature_importances_, index = config_3_feature_columns)))

