"""
in terminal run: streamlit run main.py
in another terminal run: mlflow ui
"""

import time
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
import pycaret.regression as pc_rg
import mlflow
import shap
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa


def get_model_training_logs(n_lines=10):
    file = open('logs.log', 'r')
    lines = file.read().splitlines()
    file.close()
    return lines[-n_lines:]


RG_MODEL_LIST = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'par', 'huber', 'knn', 'dt', 'et',
                 'ada', 'gbr', 'rf', 'xgboost', 'lightgbm', 'catboost', 'dummy']
RG_MODEL_LIST_1= ['knn', 'dummy','et']
RG_MODEL_LIST_2= ['rf', 'xgboost', 'lightgbm', 'dt', 'et', 'catboost']

def list_files(directory, extension):
    # list certain extension files in the folder
    return [f for f in os.listdir(directory) if f.endswith('.' + extension)]


def concat_file_path(file_folder, file_selected):
    # handle the folder path with '/' or 'without './'
    # and concat folder path and file path
    if str(file_folder)[-1] != '/':
        file_selected_path = file_folder + '/' + file_selected
    else:
        file_selected_path = file_folder + file_selected
    return file_selected_path


@st.cache_data()
def load_csv(file_selected_path, nrows):
    # load certain rows
    try:
        if nrows == -1:
            df = pd.read_csv(file_selected_path)
        else:
            df = pd.read_csv(file_selected_path, nrows=nrows)
    except Exception as ex:
        df = pd.DataFrame([])
        st.exception(ex)
    return df






def app_main():
    global tuned_model
    st.title("Analysis System of E-waste Pollutants")
    st.sidebar.image("LOGO.png")
    st.sidebar.header('Produced by :blue[_Zhang_] Laboratory :sunglasses:')
    st.sidebar.caption("Email: zhangt47@mail.sysu.edu.cn")
    st.sidebar.text("")
    st.sidebar.text("")

    if st.sidebar.checkbox('Step 1. Import complete dataset'):
        file_folder = st.sidebar.text_input('File folder of origin dataset', value="origin data")
        data_file_list = list_files(file_folder, 'csv')
        if len(data_file_list) == 0:
            st.warning(f'No file available in this folder')
        else:
            file_selected = st.sidebar.selectbox(
                'Choose file', data_file_list)
            file_selected_path = concat_file_path(file_folder, file_selected)
            st.info(f'❗Selected dataset (origin): {file_selected_path} ')
    else:
        file_selected_path = None
        st.warning(f'No origin dataset')

    if st.sidebar.checkbox('Step 2. Exploratory data analysis'):
        if file_selected_path is not None:
            if st.sidebar.button('Begin'):
                df = pd.read_csv(file_selected_path)
                pr = ProfileReport(df, explorative=True)
                st_profile_report(pr)
        else:
            st.info(f'Analysis was not possible due to unavailability of file')

    if st.sidebar.checkbox('Step 3. Machine learning'):

        if file_selected_path is not None:
            df = pd.read_csv(file_selected_path)

            global session_id, train_size, Normalize, normalize_method, target_col

            session_id = st.sidebar.number_input('Random state (reproducibility and repeatability)', value=1)
            train_size = st.sidebar.slider(label='Train size', min_value=0.5, max_value=1.0, value=0.7, step=0.05)
            Normalize = st.sidebar.checkbox('Normalize')
            normalize_method='zscore'
            if Normalize:
                normalize_method = st.sidebar.selectbox('Choose a normalize method', ['zscore','minmax','maxabs','robust'])

            try:
                cols = df.columns.to_list()
                target_col = st.sidebar.selectbox('Choose the prediction label', cols)
            except BaseException:
                st.sidebar.warning(f'The data format cannot be read correctly')
                target_col = None

            if target_col is not None and st.sidebar.button('Model selection'):

                st.success(f'3.1. Automatic data preprocessing')
                with st.spinner('Wait for it...'):
                    pc_rg.setup(
                    df,
                    target=target_col,
                    train_size=train_size,
                    normalize=Normalize,
                    normalize_method=normalize_method,
                    session_id=session_id,
                    log_experiment=True,
                    experiment_name='ml_',
                    log_plots=True,
                    verbose=False,
                    profile=True)

                st.success(f'3.2. Automatic model selection')
                st.success(f'3.2.1. All regression models awaiting assessment')
                all_models = pc_rg.models()
                st.write(all_models)

                st.success(f'3.2.2. Top five models with highest accuracy')
                with st.spinner('Wait for it...'):
                    top5 = pc_rg.compare_models(n_select=5)
                st.write(top5)

            model = st.sidebar.selectbox('Determine the model used', RG_MODEL_LIST)

            if model is not None and st.sidebar.button('Model training'):

                with st.spinner('Wait for it...'):
                    pc_rg.setup(
                    df,
                    target=target_col,
                    log_experiment=True,
                    train_size=train_size,
                    normalize=Normalize,
                    normalize_method=normalize_method,
                    session_id=session_id,
                    experiment_name='ml_',
                    log_plots=True,
                    verbose=False,
                    profile=True)
                st.success(f'3.1. Automatic data preprocessing -- Finished')

                with st.spinner('Wait for it...'):
                    b_model = pc_rg.create_model(model, return_train_score=True, verbose=False)
                st.success(f'3.2. Automatic model training -- Finished')

                with st.spinner('Wait for it...'):
                    t_model, tuner = pc_rg.tune_model(b_model, return_tuner=True)
                st.success(f'3.3. Automatic model optimization -- Finished')


                # pc_rg.finalize_model(built_model)


                st.success(f'3.4. Automatic model evaluation and interpretation')
                with st.spinner('Wait for it...'):
                    st.dataframe(pc_rg.predict_model(t_model))

                with st.spinner('Wait for it...'):
                    st.write(pc_rg.plot_model(t_model, plot='residuals', display_format="streamlit"))
                st.text('Residuals Plot')
                with st.spinner('Wait for it...'):
                    st.write(pc_rg.plot_model(t_model, plot='error', display_format="streamlit"))
                st.text(' Prediction Error Plot')
                with st.spinner('Wait for it...'):
                    st.write(pc_rg.plot_model(t_model, plot='cooks', display_format="streamlit"))
                st.text('Cooks Distance Plot')
                with st.spinner('Wait for it...'):
                    st.write(pc_rg.plot_model(t_model, plot='learning', display_format="streamlit"))
                st.text('Learning Curve')
                with st.spinner('Wait for it...'):
                    st.write(pc_rg.plot_model(t_model, plot='manifold', display_format="streamlit"))
                st.text('Manifold Learning')

                if model not in RG_MODEL_LIST_1:
                    with st.spinner('Wait for it...'):
                        st.write(pc_rg.plot_model(t_model, plot='rfe', display_format="streamlit"))
                    st.text('Recursive Feature Selection')
                    with st.spinner('Wait for it...'):
                        st.write(pc_rg.plot_model(t_model, plot='feature', display_format="streamlit"))
                    st.text('Feature Importance')

                if model in RG_MODEL_LIST_2:
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    with st.spinner('Wait for it...'):
                        st.pyplot(pc_rg.interpret_model(t_model, plot='summary'))
                    st.text(f'Summary plot using SHAP')

                logs = pc_rg.get_logs(save=True)
                print(logs)


    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    try:
        all_runs = mlflow.search_runs(experiment_ids=all_experiments)  # filter = "attributes.status = "FINISHED"",
    except:
        all_runs = []

    if len(all_runs) != 0:
        if st.sidebar.checkbox('Step 4. Model information'):
            st.text('Information of all built models:')
            st.dataframe(all_runs)
            st.success(f'To view the details:')
            ml_logs_1 = ('- enter :blue[_mlflow ui_] at the command line of the current directory')
            ml_logs_2 = ('- or run the :blue[_mlflow_] and click http://127.0.0.1:5000')
            st.markdown(ml_logs_1)
            st.markdown(ml_logs_2)
            st.text('')
            st.success(f'Important information of the selected model:')
            selected_run_id = st.sidebar.selectbox('All optimized models',
                                                   all_runs[all_runs['tags.Source'] == 'tune_model'][
                                                       'run_id'].tolist())
            selected_run_info = all_runs[(
                    all_runs['run_id'] == selected_run_id)].iloc[0, :]
            st.table(selected_run_info)


            Feature_Importance_uri = f'mlruns/1/' + selected_run_id + '/artifacts/Feature Importance.png'
            Prediction_Error_uri=f'mlruns/1/' + selected_run_id + '/artifacts/Prediction Error.png'
            Residuals_uri=f'mlruns/1/' + selected_run_id + '/artifacts/Residuals.png'
            st.image(Feature_Importance_uri)
            st.image(Prediction_Error_uri)
            st.image(Residuals_uri)


        if st.sidebar.checkbox('Step 5. Data prediction'):
            selected_run_id = st.sidebar.selectbox('Select an optimized model',
                                                   all_runs[all_runs['tags.Source'] == 'tune_model'][
                                                       'run_id'].tolist())
            selected_run_info = all_runs[(
                    all_runs['run_id'] == selected_run_id)].iloc[0, :]
            st.code(selected_run_info)


            file_folder = st.sidebar.text_input('File folder of new dataset', value="new data")
            data_file_list = list_files(file_folder, 'csv')
            if len(data_file_list) == 0:
                st.sidebar.warning(f'No file available in this folder')
            else:
                file_selected = st.sidebar.selectbox(
                        'Choose file', data_file_list)
                file_selected_path = concat_file_path(file_folder, file_selected)

                st.info(f'❗Selected dataset (new): {file_selected_path} ')


            if st.sidebar.button('Predict'):
                model_uri = f'runs:/' + selected_run_id + '/model/'
                model_loaded = mlflow.sklearn.load_model(model_uri)


                df = pd.read_csv(file_selected_path)
                st.dataframe(df)
                st.success(f'Model-predicted outcomes')
                pred = model_loaded.predict(df)
                pred_df = pd.DataFrame(pred, columns=['Predicted value'])
                st.dataframe(pred_df)
                pred_df.plot()


    else:
        st.sidebar.warning('No trained model')


if __name__ == '__main__':
    app_main()

