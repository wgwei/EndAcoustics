import pandas as pd
import numpy as np
import os

os.chdir(r'C:\Users\WG\Documents\Research\EndAcoustics')

def LAeq5min_to_LAeq1h_sigicom():
    # load data and convert the column to time stamp
    df = pd.read_csv('Data\\NV Hebburn mine water Interval list.csv')
    df['Date_time_stamp'] = pd.to_datetime(df['Date_time'], dayfirst=True)
    print(df.tail())
    start_date = df['Date_time_stamp'][0]
    end_date = df['Date_time_stamp'][df.shape[0]-1]

    current_date = start_date

    output_df = pd.DataFrame({'ID', range(24)})

    hours, LAeq_1h = [], []
    # loop each day, and hour
    while current_date <= end_date:
        
        for h in range(24): 
            start_hour = current_date + pd.Timedelta(hours=h)
            if h<23:
                end_hour = current_date + pd.Timedelta(hours=h+1)
                df2 = df[(df['Date_time_stamp']>= start_hour) & (df['Date_time_stamp']< end_hour)]
            if h==23:
                end_hour = current_date + pd.Timedelta(days=1)
                df2 = df[(df['Date_time_stamp']>= start_hour) & (df['Date_time_stamp']< end_hour)]
            dB_LAeq = df2['dB_Laeq'].values
            LAeq_1h.append(10*np.log10(np.mean(10.**(dB_LAeq/10))))
            hours.append(start_hour)
            # print(df2)
        current_date = current_date + pd.Timedelta(days=1)
    
    output_df = pd.DataFrame({'Date time': hours, 'dB_LAeq1h': LAeq_1h})
    print(output_df.head())
    print(output_df.tail())


if __name__=='__main__':
    LAeq5min_to_LAeq1h_sigicom()