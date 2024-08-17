from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from features_handlers import period_to_features

from common_functions import data_to_period


def features_to_df(data: dict) -> pd.DataFrame:
    """
    Convert features dictionary into a DataFrame
    :param data: dict: data
    :return: pd.DataFrame: data
    """
    features_dict = {pat_id: pat_data['features_dict'] for pat_id, pat_data in data.items()}
    df = pd.DataFrame.from_dict(features_dict, orient='index')
    df.index.name = 'patient_id'

    return df


def read_summary_data(path=None) -> pd.DataFrame:
    """
    Read the data from the directory
    :param path: str: path to the file
    :return: dict: data
    """
    if path is None:
        path = Path('PPG-BP dataset.csv').resolve()

    patients_data = pd.read_csv(Path(path),
                                sep="\t",
                                skiprows=1,
                                # header=None,
                                )

    patients_data.set_index('subject_ID', inplace=True)

    # print(patients_data.head())
    # print(patients_data.shape)
    # print(patients_data.columns)

    return patients_data


def read_all_data(path_data_dir=None,
                  path_patients_summary=None,
                  limit_to_n_persons=None,
                  ) -> dict:

    data_parts = defaultdict(pd.DataFrame)
    data = {}

    if path_data_dir is None:
        current_dir = Path().resolve()
        path_data_dir = Path(current_dir, 'data').resolve()

    # print('========================================================')

    patients_data = read_summary_data(path=path_patients_summary)
    # Read PPG signals:

    print('Data dir:', path_data_dir)
    files = sorted(list(path_data_dir.glob("*.txt")))

    # print(*files, sep='\n')

    # Limit the number of files to read
    limit_to_n_persons = limit_to_n_persons if (limit_to_n_persons is not None and isinstance(limit_to_n_persons, int)) else len(files // 3)
    slice_files = slice(None, limit_to_n_persons * 3)
    files = files[slice_files]

    for path in tqdm(files):

        patient_id_parts = path.stem.split('_')
        patient_id = patient_id_parts[0]
        signal_id = patient_id_parts[1]

        data_parts[patient_id] = pd.concat([data_parts[patient_id],
                                            pd.read_csv(path,
                                                        sep="\t",
                                                        header=None,
                                                        ).T
                                            .dropna()
                                            .rename(columns={0: f'PPG_{patient_id}_{signal_id}'}),
                                            ],
                                           axis=1,
                                           )

    for patient_id in data_parts.keys():
        # get systolic and diastolic blood pressures from the dataset:
        sysBP = patients_data.loc[int(patient_id), 'Systolic Blood Pressure(mmHg)']
        diaBP = patients_data.loc[int(patient_id), 'Diastolic Blood Pressure(mmHg)']
        gender = patients_data.loc[int(patient_id), 'Sex(M/F)']
        bmi = patients_data.loc[int(patient_id), 'BMI(kg/m^2)']
        age = patients_data.loc[int(patient_id), 'Age(year)']
        #hr  = patients_data.loc[int(patient_id), 'Heart Rate(b/m)']
        #hypertension = patients_data.loc[int(patient_id), 'Hypertension']
      
        avg_period = data_to_period(data_parts[patient_id])

        # Add PPG signals, systolic and diastolic blood pressures
        # into the data dictionary:
        data[patient_id] = {}
        data[patient_id]['data'] = data_parts[patient_id]
        data[patient_id]['avg_period'] = avg_period
        data[patient_id]['features_dict'] = period_to_features(avg_period)
        data[patient_id]['features_dict']['sysBP'] = sysBP
        data[patient_id]['features_dict']['diaBP'] = diaBP
        data[patient_id]['features_dict']['age'] = age
        data[patient_id]['features_dict']['gender'] = gender
        data[patient_id]['features_dict']['BMI'] = bmi
        #data[patient_id]['features_dict']['heartRate'] = hr
        #data[patient_id]['features_dict']['hypertension'] = hypertension
        

    return data
