import os
import pandas as pd

def find_xl2_measurements(current_path=os.getcwd()):
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
    return duration_in_hour



def convert_string_to_time(time_string):
    """ convert the string style input to pandas date time object
    time_string = NTi format input sucha as Start:          	2020-10-08, 12:58:56"""
    dt = time_string.strip("\n").split("\t")[-1]
    dt2 = dt.replace(",", "")
    dt_obj = pd.to_datetime(dt2)
    return dt_obj


def main():
    os.chdir(r"G:\Shared drives\9000 - 9999")
    save_dire = "C:\Users\WG\Documents\Research\EndAcoustics\Data"
    save_to_file = "D48_9000-9999.csv"
    file_path = find_xl2_measurements(current_path=os.getcwd())
    file_path.to_csv(os.path.join(save_dire, save_to_file))

if __name__=="__main__":
    main()