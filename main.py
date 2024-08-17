
from pathlib import Path

from io_functions import features_to_df, read_all_data


if __name__ == '__main__':

    data = read_all_data(path_data_dir=Path('data'),
                         path_patients_summary=Path('PPG-BP dataset.csv'),
                         limit_to_n_persons=3,
                         )

    features = features_to_df(data)

    # ============================================================
    # printing results

    print('==========================================================================')
    print('Data of the first patient in the dictionary:')
    print('--------------------------------------------------------------------------')
    print(data[list(data.keys())[0]])
    print('==========================================================================')

    # Convert all patients features into features' DataFrame

    print('==========================================================================')
    print('All patients Features:')
    print('--------------------------------------------------------------------------')
    print(features)
    print('==========================================================================')
