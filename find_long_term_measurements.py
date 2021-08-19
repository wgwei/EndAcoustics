import os
import pandas as pd
import numpy as np

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
                    print(line_time, splited_line[index_max], splited_line[index_eq])
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

def get_night_time(df):
    """ df = ["Date", "Time", "LAFmax_dt", "LAeq_dt", "Datetime_obj"]
        the last column of df should be Datetime object
    """
    start_date = df.iloc[0, 4].date()
    start_date = pd.to_datetime(start_date)  # convert the date only object to datetime object
    mid_date = start_date + pd.to_timedelta("1 days")
    end_date = start_date + pd.to_timedelta("2 days")

    night_start_1 = start_date + pd.to_timedelta("23 hours")
    night_end_1 = mid_date + pd.to_timedelta("7 hours")
    night_start_2 = mid_date + pd.to_timedelta("23 hours")
    night_end_2 = end_date + pd.to_timedelta("7 hours")

    return [night_start_1, night_end_1, night_start_2, night_end_2]


def calc_LAeq8hr_LAFmax(df):
    """ df = data frame of ["Date", "Time", "LAFmax_dt", "LAeq_dt"]
        to calcualted the LAeq8hr and LAFmax every 1min,  5min, and 15min
    """
    df = df.dropna()  # after the basic data there are some descritpions, remove them from the data sapce
    df = df.iloc[0:-1, :] # remove the last row as NAT
    datetime_string = [str(d) + " " + str(t) for d, t in zip(df["Date"], df["Time"])]
    datetime_obj = [pd.to_datetime(dt) for dt in datetime_string]
    df["Datetime"] = pd.DataFrame({"Datetime_obj":datetime_obj})

    [night_start_1, night_end_1, night_start_2, night_end_2] = get_night_time(df)

    df_night = df[(df["Datetime"]>=night_start_1) & (df["Datetime"]<night_end_1)]
    
    return df_night


def main2():
    """ Extract the night time data in second resolution"""
    os.chdir(r"C:\Users\WG\Documents\Research\EndAcoustics")
    directories = []
    filename = "Data\\Orange_recovery2_KT_2020-10-08_SLM_000_123_Log.txt"
    df2 = extract_LAeq_LAFmax_dt2(filename)
    df2.to_csv("Nighttime_data.csv")
    print(df2.head())
    print(df2.tail())


if __name__=="__main__":
    main2()