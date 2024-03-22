from datetime import datetime, timedelta
from pathlib import Path

from pandas import read_excel, concat


def concat_weather():
    path_root = Path(__file__).parent
    df_ls = []
    for skip_rows, c, file_name in ((8, [], 'Weather Canopy & Soil Temperatures Energy Balance 1993 hourly.ods'),
                                    (8, [], 'Weather Canopy & Soil Temperatures Energy Balance 1994 hourly.ods'),
                                    (76, [], 'Weather 1996 hourly.ods'),
                                    (68, ['SHADO'], 'Weather 1997 hourly.ods')):
        weather_df = read_excel(
            path_root.parent / 'data_raw' / file_name, engine='odf', sheet_name='Sheet1', skiprows=skip_rows,
            usecols=['YEAR', 'DOY', 'HOUR', 'SRAD', 'TDRY', 'RAIN', 'TDEW', 'WIND', 'PARD', 'TWET', 'TSOIL'] + c)
        weather_df.dropna(inplace=True)
        df_ls.append(weather_df)

    df = concat(df_ls, ignore_index=True)
    df = df[df['PARD'] >= 0]
    df.loc[:, 'DATE'] = df.apply(
        lambda x: datetime(int(x['YEAR']) - 1, 12, 31) + timedelta(days=x['DOY'], hours=x['HOUR'] - 1) + timedelta(
            hours=1), axis=1)
    path_output = path_root.parent / 'data_fmt/weather.csv'
    with open(path_output, mode='w') as f:
        f.write(f"# This file was automatically generated using:\n"
                f"# ~/{'/'.join([s for s in Path(__file__).parts[-4:]])}\n"
                f"#\n")
    df.to_csv(path_output, sep=';', decimal='.', mode='a', index=False)
    pass


if __name__ == '__main__':
    concat_weather()
