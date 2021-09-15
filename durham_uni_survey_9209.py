import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

os.chdir(r'G:\Shared drives\9000 - 9999\9200 - 9299\9209 Department of Chemistry, Durham University\Calculations\210908 Evening Survey\Data')

def split_data(line):
    """ split the line (string) to line = ['', '2021-06-03', '10:35:31', '00:00:01', '46.8', '46.3', '46.7',...]
        return the splitted list
    """
    line = line.strip("\n").replace(" ", "")
    line = line.split("\t") # Positon 1 is date, position 2 is time, position 6  is LAFmax_dt, positon 10 s LAeq_dt
    if len(line)>5:
        return line
    else:
        return []


def get_filenames():
    ''' get the file names with the summary of LAeq and 1/3 octave band information
        these files end with RTA_3rd_Report.txt
    '''
    all_filenames = os.listdir()
    filenames = []
    for f in all_filenames:
        if 'RTA_3rd_Report.txt' in f:
            filenames.append(f)
    return filenames


def export_LAeq_thirdoctave(filename):
    print(filename)
    with open(filename) as fn:
        lines = fn.readlines()
        for line in lines:
            if 'LAeq' in line:
                line_values = split_data(line)  #['', 'LAeq', '-32.2', '-24.5', '-15.1', '-4.7', ...] 
                line_values = [float(v) for v in line_values[2::]]  # romve the empty strng and 'LAeq', keep the values only
                LAeq = 10*np.log10(sum(10**(np.array(line_values)/10)))  # calc the LAeq
                line_values.insert(0, LAeq)
                line_values.insert(0, filename)
                print(line_values)
    return line_values


def main():
    ''' create a data frame to include file name, LAeq, 1/3 octave band from 6.3 hz to 20k Hz'''
    data_cols = ['Filename', 'LAeq', 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0]
    df = pd.DataFrame(columns=data_cols)
    filenames = get_filenames()
    for f in filenames:
        line_values = export_LAeq_thirdoctave(f)
        df = df.append(pd.DataFrame([line_values], columns=data_cols))
    df.to_csv('Filename_LAeq_third_octave_summary.csv')


if __name__=='__main__':
    main()