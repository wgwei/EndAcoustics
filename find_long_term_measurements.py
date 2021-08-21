import csv
import os
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def find_xl2_measurements_48hrs(current_path=os.getcwd()):
    """ walk the job folders and find the measurements undertaken by XL2
        only duration >=48 hours are included  
    """
    directories, files = [], []
    folders = os.walk(current_path)
    y = [x[0] for x in folders]
    for yy in y:
        print("----------------------------")
        print(yy)
        os.chdir(yy)
        dires = os.listdir(".")
        for fname in dires:
            if ("RTA_3rd_Report.txt" in fname) or ("RTA_Oct_Report.txt" in fname):
                print(fname)
                duration_in_hour = extract_survey_duration(fname)
                if duration_in_hour>=48:
                    directories.append(yy)
                    files.append(fname)
    path_file = pd.DataFrame({"Directories": directories, "File_name": files})
    print(path_file.head())
    return path_file


def extract_survey_duration(filename):
    """ Scan each line of the report file and find the start and end time of the noise measuerment """
    with open(filename) as report:
        try:
            contents = report.readlines()
            for c in contents:
                if "Start" in c:
                    start_time = convert_string_to_time(c)
                    print("Start time: ", start_time)
                if "End" in c:
                    end_time = convert_string_to_time(c)
                    print("End time: ", end_time)
                    a = end_time - start_time
                    duration_in_hour = a / pd.Timedelta('1 hour')
                    break
        except:
            duration_in_hour = 0
    return duration_in_hour


def convert_string_to_time(time_string):
    """ convert the string style input to pandas date time object
    time_string = NTi format input sucha as Start:          	2020-10-08, 12:58:56"""
    dt = time_string.strip("\n").split("\t")[-1]
    dt2 = dt.replace(",", "")  # dt2="2020-10-08 12:52:45"
    dt_obj = pd.to_datetime(dt2)
    return dt_obj


def main():
    """ walk the job folders and find the survey which have more than 48 hour data"""
    folders = [r"G:\Shared drives\6000 - 6999", r"G:\Shared drives\5000 - 5999", r"G:\Shared drives\4000 - 4999"]
    save_names = [r"D48_6000-6999.csv", r"D48_5000-5999.csv", r"D48_4000-4999.csv"]
    save_dire = r"C:\Users\WG\Documents\Research\EndAcoustics\Data"
    for v in range(len(folders)):
        os.chdir(folders[v])
        print(folders[v])
        file_path = find_xl2_measurements_48hrs(current_path=os.getcwd())
        save_to_file = save_names[v]
        save_loc = os.path.join(save_dire, save_to_file)
        file_path.to_csv(os.path.join(save_dire, save_loc))


def extract_LAeq_LAFmax_dt2(filename):
    """ extract the LAeq,8hr and LAFmax in 1 min, 5min
        report and plot 1st, 5th, 10th 20th LAFmax
        NOTE: pandas took too much time to load such a big file
        therefore, the data is read line by line and processed.
    """
    # find the which row the data started to log
    # fine the index of the LAmax_dt and LAeq_dt
    with open(filename) as fn:
        row_n = 0
        for line in fn:
            row_n += 1
            if "LAFmax_dt" in line:
                splited_line = split_data(line)
                for n, elem in enumerate(splited_line):
                    if "LAFmax_dt" in elem:
                        index_max = n
                    if "LAeq_dt" in elem:
                        index_eq = n
                break  # find the number of comment rows by finding LAFmax_dt. stop the loop
    
    # read the file again, skip 5 extra row to make sure read valid data
    date_time, LAFmax_dt, LAeq_dt = [], [], []
    with open(filename) as fn2:
        for _ in range(row_n+5):  # skip row_+5 rows
            next(fn2)
        for line in fn2:
            splited_line = split_data(line)
            if len(splited_line)>0:  # to find the last valid row of data. after that there is an empty row
                line_time, night = is_nighttime_data(splited_line)
                if night==1:
                    # print(line_time, splited_line[index_max], splited_line[index_eq])
                    date_time.append(line_time)
                    LAFmax_dt.append(splited_line[index_max])
                    LAeq_dt.append(splited_line[index_eq])
            else:
                break  # this break avoid load invalid data after the final row
    df = pd.DataFrame({"Date_time":date_time, "LAFmax_dt":LAFmax_dt, "LAeq_dt":LAeq_dt})
    return df


def split_data(line):
    """ split the line (string) to line = ['', '2021-06-03', '10:35:31', '00:00:01', '46.8', '46.3', '46.7',...]
        return the splitted list
    """
    line = line.strip("\n").replace(" ", "")
    line = line.split("\t") # Positon 1 is date, position 2 is time, position 6  is LAFmax_dt, positon 10 s LAeq_dt
    if len(line)>10:
        return line
    else:
        return []


def is_nighttime_data(line):
    """ line = ['', '2021-06-03', '10:35:31', '00:00:01', '46.8', '46.3', '46.7',...]
        it is the splitted line data in string format
    """
    line_datetime = pd.to_datetime(line[1] + " " + line[2])
    line_hour = line_datetime.hour
    if line_hour>=23 or line_hour<7:
        night = 1
    else:
        night = 0
    return line_datetime, night


def main2():
    """ Extract the night time data in second resolution"""
    os.chdir(r"C:\Users\WG\Documents\Research\EndAcoustics")
    # filename = "Data\\Orange_recovery2_KT_2020-10-08_SLM_000_123_Log.txt"
    # df2 = extract_LAeq_LAFmax_dt2(filename)
    # df2.to_csv("Nighttime_data.csv")
    # print(df2.head())
    # print(df2.tail())
    directory_jobs = ['D48_9000-9999.csv', 'D48_8000-8999.csv', "D48_7000-7999.csv", "D48_6000-6999.csv", "D48_5000-5999.csv", "D48_4000-4999.csv"]
    for dir_f in directory_jobs:
        paths = pd.read_csv("Data\\" + dir_f)    
        for n, p in enumerate(paths["Directories"]):
            csv_output_filename = str(n) + "__" + p.split("\\")[4] + ".csv" # position 4 is the job name, will generate 0__9653 Station road, noise assessment style name
            try:
                all_files = os.listdir(p)
                for f123 in all_files:
                    if "_123_Log.txt" in f123:
                        data_file = p + "\\" + f123
                        print(data_file)
                        print(csv_output_filename)
                        try:
                            df2 = extract_LAeq_LAFmax_dt2(data_file)
                            df2.to_csv(csv_output_filename)
                        except:
                            print("FAILED TO LOAD FILE")
            except:
                print("FILE NOT FOUND")


def get_night_time(start_date_time):
    """ start_date_time = "2021-08-01 12:32:54" a string
        the last column of df should be Datetime object
    """
    start_date_time = pd.to_datetime(start_date_time)
    start_date = start_date_time.date()
    start_date = pd.to_datetime(start_date)  # convert the date only object to datetime object
    print(start_date)
    mid_date = start_date + pd.to_timedelta("1 days")
    end_date = start_date + pd.to_timedelta("2 days")

    night_start_1 = start_date + pd.to_timedelta("23 hours")
    night_end_1 = mid_date + pd.to_timedelta("7 hours")
    night_start_2 = mid_date + pd.to_timedelta("23 hours")
    night_end_2 = end_date + pd.to_timedelta("7 hours")
    print([night_start_1, night_end_1, night_start_2, night_end_2])
    return [night_start_1, night_end_1, night_start_2, night_end_2]


def calc_LAFmax_interval_in_sec(df, step, which_day=0):
    """ df = dataFrame including["index", "Date_time", "LAFmax_dt", "LAeq_dt"]
        step = in seconds, collect data every step seconds, for example every 60 seconds
        which_day = 0 or 1. 0 for the first day data, 1 for the second day data
    """
    sec_in_8hr = 60*60*8  # how many secons in 8 hours
    LAFmax_1interval = []
    for m in range(int(sec_in_8hr/step)):
        ss = m * step + sec_in_8hr*which_day
        es = (m + 1) * step + sec_in_8hr*which_day
        data_1unit = df["LAFmax_dt"][ss:es].values
        LAFmax_1interval.append(max(data_1unit))
    return LAFmax_1interval


def get_nth_LAFmax(LAFmax):
    """ get 1st 5th 10th 20th LAFmax
        LAFmax = [list of LAFmax in different intervals]
    """
    LAFmax.sort(reverse=True)  # big number first. 
    LAFmax_1 = LAFmax[0]
    LAFmax_5 = LAFmax[4]
    LAFmax_10 = LAFmax[9]
    LAFmax_20 = LAFmax[19]
    return [LAFmax_1, LAFmax_5, LAFmax_10, LAFmax_20]


def calc_LAeq8hr_LAFmax(filename):
    """ filename = file contains only nighttime data ["index", "Date_time", "LAFmax_dt", "LAeq_dt"] the resolution is most likely to be 1 second
        Caclulate the LAeq8hr and LAFmax every 1min,  5min, and 15min
        NOTE: as it takes too long to handle date time object. here use default 1 sec to count the time. 
        it is assume the sampling rate of the measurement is 1 sec. ths is also the default setting of NTi meter
    """
    
    try:
        if '4233' in filename:  # the samepling is not one data per sec for this job
            is_df_valid = False
        else:
            df = pd.read_csv(filename)
            if df.shape[0] > 1000:  # valid if more than 1000 lines recorded
                is_df_valid = True
            else:
                is_df_valid = False    
    except:
        print('FILE NOT FOUND')
        is_df_valid = False

    if is_df_valid:
        # date_obj = [pd.to_datetime(x) for x in df["Date_time"]]
        # df["Date_obj"] = pd.Series(date_obj)

        # Calcualte LAeq,8hr
        sec_in_8hr = 60*60*8  # how many secons in 8 hours
        LAeq_1s_8hr_day1 = np.array(df["LAeq_dt"][0:sec_in_8hr].values)
        LAeq_8hr_day1 = 10.*np.log10(mean(10**(LAeq_1s_8hr_day1/10)))
        LAeq_1s_8hr_day2 = np.array(df["LAeq_dt"][sec_in_8hr:sec_in_8hr*2].values)
        LAeq_8hr_day2 = 10.*np.log10(mean(10**(LAeq_1s_8hr_day2/10)))

        step_1 = 60  # how many seconds in one minute
        step_5 = 60*5  # seconds in 5 minute
        step_15 = 60*15  # secnods in 15 minutes

        # day 1
        LAFmax_1min_day1 = calc_LAFmax_interval_in_sec(df, step_1, 0)  # 1 minute LAFmax
        LAFmax_5min_day1 = calc_LAFmax_interval_in_sec(df, step_5, 0)  # 5 minutes interval LAFmax
        LAFmax_15min_day1 = calc_LAFmax_interval_in_sec(df, step_15, 0)  # 15 minutes interval LAFmax
        [LAFmax_1_1min_day1, LAFmax_5_1min_day1, LAFmax_10_1min_day1, LAFmax_20_1min_day1] = get_nth_LAFmax(LAFmax_1min_day1)
        [LAFmax_1_5min_day1, LAFmax_5_5min_day1, LAFmax_10_5min_day1, LAFmax_20_5min_day1] = get_nth_LAFmax(LAFmax_5min_day1)
        [LAFmax_1_15min_day1, LAFmax_5_15min_day1, LAFmax_10_15min_day1, LAFmax_20_15min_day1] = get_nth_LAFmax(LAFmax_15min_day1)
        
        # day 2
        LAFmax_1min_day2 = calc_LAFmax_interval_in_sec(df, step_1, 1)  # 1 minute LAFmax
        LAFmax_5min_day2 = calc_LAFmax_interval_in_sec(df, step_5, 1)  # 5 minutes interval LAFmax
        LAFmax_15min_day2 = calc_LAFmax_interval_in_sec(df, step_15, 1)  # 15 minutes interval LAFmax
        [LAFmax_1_1min_day2, LAFmax_5_1min_day2, LAFmax_10_1min_day2, LAFmax_20_1min_day2] = get_nth_LAFmax(LAFmax_1min_day2)
        [LAFmax_1_5min_day2, LAFmax_5_5min_day2, LAFmax_10_5min_day2, LAFmax_20_5min_day2] = get_nth_LAFmax(LAFmax_5min_day2)
        [LAFmax_1_15min_day2, LAFmax_5_15min_day2, LAFmax_10_15min_day2, LAFmax_20_15min_day2] = get_nth_LAFmax(LAFmax_15min_day2)
        
        return [LAeq_8hr_day1, 
        LAFmax_1_1min_day1, LAFmax_5_1min_day1, LAFmax_10_1min_day1, LAFmax_20_1min_day1,
        LAFmax_1_5min_day1, LAFmax_5_5min_day1, LAFmax_10_5min_day1, LAFmax_20_5min_day1, 
        LAFmax_1_15min_day1, LAFmax_5_15min_day1, LAFmax_10_15min_day1, LAFmax_20_15min_day1, 
        LAeq_8hr_day2,
        LAFmax_1_1min_day2, LAFmax_5_1min_day2, LAFmax_10_1min_day2, LAFmax_20_1min_day2, 
        LAFmax_1_5min_day2, LAFmax_5_5min_day2, LAFmax_10_5min_day2, LAFmax_20_5min_day2, 
        LAFmax_1_15min_day2, LAFmax_5_15min_day2, LAFmax_10_15min_day2, LAFmax_20_15min_day2]
    else:
        return [0]*26


def put_file_together():
    ''' Put all files together'''
    directory_jobs = ["D48_9000-9999.csv", "D48_8000-8999.csv", "D48_7000-7999.csv", "D48_6000-6999.csv", "D48_5000-5999.csv", "D48_4000-4999.csv"]
    ref_path, csv_out_filenames = [], []
    for dir_f in directory_jobs:
        paths = pd.read_csv("Data\\" + dir_f)
        for n, p in enumerate(paths["Directories"]):
            ref_path.append(p)
            csv_output = str(n) + "__" + p.split("\\")[4] + ".csv" # position 4 is the job name, will generate 0__9653 Station road, noise assessment style name
            csv_out_filenames.append(csv_output)
    return ref_path, csv_out_filenames


def main3():
    """ calculate Laeq8hr, LAFmax,1min interval 5min interval. """
    ref_path, csv_out_filenames = put_file_together()
    data_cols = ['Directory', 'LAeq_8hr_day1', 
    'LAFmax_1_1min_day1', 'LAFmax_5_1min_day1', 'LAFmax_10_1min_day1', 'LAFmax_20_1min_day1',
    'LAFmax_1_5min_day1', 'LAFmax_5_5min_day1', 'LAFmax_10_5min_day1', 'LAFmax_20_5min_day1', 
    'LAFmax_1_15min_day1', 'LAFmax_5_15min_day1', 'LAFmax_10_15min_day1', 'LAFmax_20_15min_day1', 
    'LAeq_8hr_day2',
    'LAFmax_1_1min_day2', 'LAFmax_5_1min_day2', 'LAFmax_10_1min_day2', 'LAFmax_20_1min_day2', 
    'LAFmax_1_5min_day2', 'LAFmax_5_5min_day2', 'LAFmax_10_5min_day2', 'LAFmax_20_5min_day2', 
    'LAFmax_1_15min_day2', 'LAFmax_5_15min_day2', 'LAFmax_10_15min_day2', 'LAFmax_20_15min_day2']
    df = pd.DataFrame(columns=data_cols)
    for file_path, data_1s_file in zip(ref_path, csv_out_filenames):
        print(data_1s_file)
        result_list = calc_LAeq8hr_LAFmax(data_1s_file)
        row = [[file_path] + result_list]   # [[path, LAeq8hr, LAFmax....]]
        df = df.append(pd.DataFrame(row, columns=data_cols))
    df.to_csv('Summary_table.csv')


def plot_output():
    summary = pd.read_csv('Summary_table.csv')
    summary = summary[summary['LAeq_8hr_day1']>0]  # remove the rows only have zeros

    # plot LAeq_8hr_day1 minus LAeq_8hr_day2
    Ld8hr = summary['LAeq_8hr_day1'] - summary['LAeq_8hr_day2']
    plt.boxplot(Ld8hr)
    plt.grid()

    # plot LAFmax_1, 5, 10, 20 difference based on different time intervals 1 minute and 5 minute
    LAF_5 = list(summary['LAFmax_5_1min_day1'].values - summary['LAFmax_5_5min_day1'].values)
    LAF_5 = LAF_5 + list(summary['LAFmax_5_1min_day2'].values - summary['LAFmax_5_5min_day2'].values)  # merge day 1 and day 2 data
    LAF_10 = list(summary['LAFmax_10_1min_day1'].values - summary['LAFmax_10_5min_day1'].values)
    LAF_10 = LAF_10 + list(summary['LAFmax_10_1min_day2'].values - summary['LAFmax_10_5min_day2'].values)
    LAF_20 = list(summary['LAFmax_20_1min_day1'].values - summary['LAFmax_20_5min_day1'].values)
    LAF_20 = LAF_20 + list(summary['LAFmax_20_1min_day2'].values - summary['LAFmax_20_5min_day2'].values)
    LAFmax_resoluton = [LAF_5, LAF_10, LAF_20]
    plt.figure()
    plt.boxplot(LAFmax_resoluton)
    plt.grid()
    plt.ylabel('LAFmax 1min - LAFmax 5min, dB')
    plt.xticks([1, 2, 3], ["5th LAFmax", "10th LAFmax", "20th LAFmax"])
    plt.title("Difference between 1 minute and 5-minute resolution")

    # 1 minute resolution LAFmax_1, LAFmax_5, LAFmax_10, LAFmax_20 minus LAeq_8hr
    LAFmax1_LAeq8h = list(summary['LAFmax_1_1min_day1'] - summary['LAeq_8hr_day1'])
    LAFmax1_LAeq8h = LAFmax1_LAeq8h + list(summary['LAFmax_1_1min_day2'] - summary['LAeq_8hr_day2'])
    LAFmax5_LAeq8h = list(summary['LAFmax_5_1min_day1'] - summary['LAeq_8hr_day1'])
    LAFmax5_LAeq8h = LAFmax5_LAeq8h + list(summary['LAFmax_5_1min_day2'] - summary['LAeq_8hr_day2'])
    LAFmax10_LAeq8h = list(summary['LAFmax_10_1min_day1'] - summary['LAeq_8hr_day1'])
    LAFmax10_LAeq8h = LAFmax10_LAeq8h + list(summary['LAFmax_10_1min_day2'] - summary['LAeq_8hr_day2'])
    LAFmax20_LAeq8h = list(summary['LAFmax_20_1min_day1'] - summary['LAeq_8hr_day1'])
    LAFmax20_LAeq8h = LAFmax20_LAeq8h + list(summary['LAFmax_20_1min_day2'] - summary['LAeq_8hr_day2'])
    LAFmax_1min_LAeq8h = [LAFmax1_LAeq8h, LAFmax5_LAeq8h, LAFmax10_LAeq8h, LAFmax20_LAeq8h]
    plt.figure()
    plt.boxplot(LAFmax_1min_LAeq8h)
    plt.grid()
    plt.ylabel('LAFmax 1min - LAeq_8hr, dB')
    plt.xticks([1, 2, 3, 4], ["1st LAFmax - LAeq8hr", "5th LAFmax - LAeq8hr", "10th LAFmax - LAeq8hr", "20th LAFmax - LAeq8hr"])
    plt.title("Difference between LAFmax 1min and LAeq_8hr")

    # 5 minute resolution LAFmax_1, LAFmax_5, LAFmax_10, LAFmax_20 minus LAeq_8hr
    LAFmax1_LAeq8h = list(summary['LAFmax_1_5min_day1'] - summary['LAeq_8hr_day1'])
    LAFmax1_LAeq8h = LAFmax1_LAeq8h + list(summary['LAFmax_1_5min_day2'] - summary['LAeq_8hr_day2'])
    LAFmax5_LAeq8h = list(summary['LAFmax_5_5min_day1'] - summary['LAeq_8hr_day1'])
    LAFmax5_LAeq8h = LAFmax5_LAeq8h + list(summary['LAFmax_5_5min_day2'] - summary['LAeq_8hr_day2'])
    LAFmax10_LAeq8h = list(summary['LAFmax_10_5min_day1'] - summary['LAeq_8hr_day1'])
    LAFmax10_LAeq8h = LAFmax10_LAeq8h + list(summary['LAFmax_10_5min_day2'] - summary['LAeq_8hr_day2'])
    LAFmax20_LAeq8h = list(summary['LAFmax_20_5min_day1'] - summary['LAeq_8hr_day1'])
    LAFmax20_LAeq8h = LAFmax20_LAeq8h + list(summary['LAFmax_20_5min_day2'] - summary['LAeq_8hr_day2'])
    LAFmax_5min_LAeq8h = [LAFmax1_LAeq8h, LAFmax5_LAeq8h, LAFmax10_LAeq8h, LAFmax20_LAeq8h]
    plt.figure()
    plt.boxplot(LAFmax_5min_LAeq8h)
    plt.grid()
    plt.ylabel('LAFmax 5min - LAeq_8hr, dB')
    plt.xticks([1, 2, 3, 4], ["1st LAFmax - LAeq8hr", "5th LAFmax - LAeq8hr", "10th LAFmax - LAeq8hr", "20th LAFmax - LAeq8hr"])
    plt.title("Difference between LAFmax 5min and LAeq_8hr")


if __name__=="__main__":
    plot_output()
    plt.show()