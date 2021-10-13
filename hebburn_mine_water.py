import csv
from datetime import datetime
import os
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pandas.core.tools.datetimes import to_datetime

def to_ymd_hms(time_string):
    ''' convert the time string 30/07/2021  18:00:00 to 2021-07-30 18:00:00 style'''
    s = time_string.split(' ')
    hms = s[-1]
    # dmy = s[0].split('/')
    dmy = s[0].split('-')
    d = dmy[2]
    m = dmy[1]
    y = dmy[0]
    ymd_hms = str(y) + '-' + str(m) + '-' + str(d) + ' ' + hms
    return ymd_hms  # a string '2021-07-30 18:00:00'


class Noise():
    def __init__(self,noise_file):
        print('Load noise data')
        self.noise = pd.read_csv(noise_file, dtype={'Leq': np.float64, 'Lmax': np.float64})
        datetime_obj = []
        for dt in self.noise['Time']:
            ymd_hms = to_ymd_hms(dt)
            datetime_obj.append(pd.to_datetime(ymd_hms))
        self.noise['Time_obj'] = pd.Series(datetime_obj)
        print(self.noise.head())
        print(self.noise.tail())

        # find the start and end date
        start_date = self.noise['Time_obj'].iloc[0].date()
        self.start_date = pd.to_datetime(start_date)
        end_date = self.noise['Time_obj'].iloc[-1].date()
        end_date = pd.to_datetime(end_date)
        delta_days = end_date - self.start_date
        self.days = delta_days.days+1

        #create empty dataframe
        self.day_all = [start_date + pd.Timedelta(n, "days") for n in range(self.days)]

    def calc_LAeq0800_1800(self):
        all_data = []
        for n in range(self.days):
            day_n = self.start_date + pd.Timedelta(n, "days")
            start_time = day_n + pd.Timedelta(8, "hours")
            end_time = day_n + pd.Timedelta(18, "hours")
            df = self.noise[(self.noise['Time_obj']>=start_time) & (self.noise['Time_obj']<end_time)]
            LAeq0800_1800 = 10.*np.log10(mean(10**(df['Leq'].values/10)))
            all_data.append(LAeq0800_1800)
        output = pd.DataFrame(all_data, index=self.day_all, columns=['LAeq0800_1800'])
        output = output.transpose()
        output.to_csv('LAeq0800_1800.csv')

    def count_events(self):
        ''' count the nubmer of events when LAeq exceeds 70, 75 and 80 dBA
        '''
        all_data = []
        for n in range(self.days):
            day_n = self.start_date + pd.Timedelta(n, "days")
            start_time = day_n + pd.Timedelta(7, "hours")
            end_time = day_n + pd.Timedelta(19, "hours")
            df = self.noise[(self.noise['Time_obj']>=start_time) & (self.noise['Time_obj']<end_time)]
            count70 = df[df['Leq']>70].count()  # return [x, x, x] count for each columen in df
            count75 = df[df['Leq']>75].count()
            count80 = df[df['Leq']>80].count()
            all_data.append([count70.values[0], count75.values[0], count80.values[0]])
        output = pd.DataFrame(all_data, index=self.day_all, columns=['>70dB LAeq5min', '>75dB LAeq5min', '>80dB LAeq5min'])
        output = output.transpose()
        output.to_csv('Noise events count.csv')

    def plot_time_history(self):
        wk1_start_date = self.start_date
        wk1_end_date = wk1_start_date + pd.Timedelta(5, "days")
        wk2_start_date = wk1_start_date + pd.Timedelta(7, "days")
        wk2_end_date = wk2_start_date + pd.Timedelta(5, "days")
        wk3_start_date = wk2_start_date + pd.Timedelta(7, "days")
        wk3_end_date = wk3_start_date + pd.Timedelta(5, "days")
        wk4_start_date = wk3_start_date + pd.Timedelta(7, "days")
        wk4_end_date = wk4_start_date + pd.Timedelta(5, "days")
        weeks = [[wk1_start_date, wk1_end_date], [wk2_start_date, wk2_end_date], 
        [wk3_start_date, wk3_end_date], [wk4_start_date, wk4_end_date]]

        for wk in weeks:
            start_time = wk[0]
            end_time = wk[1]
            df2 = self.noise[(self.noise['Time_obj']>=start_time) & (self.noise['Time_obj']<end_time)]
            data_len = df2.shape[0]
            x_tick_id = [0, int(data_len*0.2), int(data_len*0.4), int(data_len*0.6), int(data_len*0.8)]
            x_tick_values = [df2['Time_obj'].iloc[i].date() for i in x_tick_id]
            plt.figure()
            plt.plot(range(data_len), df2['Leq'])
            plt.plot(range(data_len), df2['Lmax'])
            plt.grid()
            plt.ylabel('dB')
            plt.legend(['LAeq','LAFmax'])
            plt.xticks(x_tick_id, x_tick_values)
        plt.show()


class Vibration():
    def __init__(self, vib_file):
        print('Load vibration data')
        self.vib = pd.read_csv(vib_file, dtype={'V':np.float64})
        datetime_obj = []
        for dt in self.vib['Time']:
            ymd_hms = to_ymd_hms(dt)
            datetime_obj.append(pd.to_datetime(ymd_hms))
        self.vib['Time_obj'] = pd.Series(datetime_obj)
        print(self.vib.head())
        print(self.vib.tail())

        # find the start and end date
        start_date = self.vib['Time_obj'].iloc[0].date()
        self.start_date = pd.to_datetime(start_date)
        end_date = self.vib['Time_obj'].iloc[-1].date()
        end_date = pd.to_datetime(end_date)
    
    def count_events(self):
        ''' count the nubmer of events when LAeq exceeds 70, 75 and 80 dBA
        '''
        # find the start and end date
        start_date = self.vib['Time_obj'].iloc[0].date()
        start_date = pd.to_datetime(start_date)
        end_date = self.vib['Time_obj'].iloc[-1].date()
        end_date = pd.to_datetime(end_date)
        delta_days = end_date - start_date
        days = delta_days.days+1

        #create empty dataframe
        day_all = [start_date + pd.Timedelta(n, "days") for n in range(days)]

        all_data = []
        for n in range(days):
            day_n = start_date + pd.Timedelta(n, "days")
            start_time = day_n + pd.Timedelta(7, "hours")
            end_time = day_n + pd.Timedelta(19, "hours")
            df = self.vib[(self.vib['Time_obj']>=start_time) & (self.vib['Time_obj']<end_time)]
            count3 = df[df['V']>3].count()  # return [x, x, x] count for each columen in df
            count5 = df[df['V']>5].count()
            count10 = df[df['V']>10].count()
            all_data.append([count3.values[0], count5.values[0], count10.values[0]])
        output = pd.DataFrame(all_data, index=day_all, columns=['>3 mm/s', '>5 mm/s', '>10 mm/s'])
        output = output.transpose()
        output.to_csv('Vibratoin events count.csv')
    

    def plot_time_history(self):
        wk1_start_date = self.start_date
        wk1_end_date = wk1_start_date + pd.Timedelta(5, "days")
        wk2_start_date = wk1_start_date + pd.Timedelta(7, "days")
        wk2_end_date = wk2_start_date + pd.Timedelta(5, "days")
        wk3_start_date = wk2_start_date + pd.Timedelta(7, "days")
        wk3_end_date = wk3_start_date + pd.Timedelta(5, "days")
        wk4_start_date = wk3_start_date + pd.Timedelta(7, "days")
        wk4_end_date = wk4_start_date + pd.Timedelta(5, "days")
        weeks = [[wk1_start_date, wk1_end_date], [wk2_start_date, wk2_end_date], 
        [wk3_start_date, wk3_end_date], [wk4_start_date, wk4_end_date]]

        for wk in weeks:
            start_time = wk[0]
            end_time = wk[1]
            df2 = self.vib[(self.vib['Time_obj']>=start_time) & (self.vib['Time_obj']<end_time)]
            data_len = df2.shape[0]
            x_tick_id = [0, int(data_len*0.2), int(data_len*0.4), int(data_len*0.6), int(data_len*0.8)]
            x_tick_values = [df2['Time_obj'].iloc[i].date() for i in x_tick_id]
            plt.figure()
            plt.plot(range(data_len), df2['V'])
            plt.plot(range(data_len), df2['L'])
            plt.plot(range(data_len), df2['T'])
            plt.grid()
            plt.ylabel('mm/s')
            plt.legend(['V','L', 'T'])
            plt.xticks(x_tick_id, x_tick_values)
        plt.show()


def main():
    os.chdir(r"H:\Shared drives\8000 - 8999\8500 - 8599\8527 Hebburn Mine Water noise and vib monitoring\Reports\Monitoring report\20210913-20211009")
    file_noise = "interval_8527_6438361_2021-09-13-0000-2021-10-08-2359_noise.csv"
    file_vibration = "interval_8527_6438361_2021-09-13-0000-2021-10-08-2359_vibration.csv"
    # noise_obj = Noise(file_noise)
    # noise_obj.count_events()
    # noise_obj.calc_LAeq0800_1800()
    # noise_obj.plot_time_history()
    vib_obj = Vibration(file_vibration)
    # vib_obj.count_events()
    vib_obj.plot_time_history()


if __name__=="__main__":
    main()