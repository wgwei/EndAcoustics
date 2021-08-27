import numpy as np
from faq_tools import octave_bands, fft_wave_data, A_weighting
from scipy.io import wavfile
import pandas as pd
import os
import matplotlib.pylab as plt

def adjust_value_by_LAeqT():
    ''' Calcualte the adjusted value from FFT to LAFmax
    '''
    filenames = ['peak_1.wav', 'peak_2.wav', 'peak_3.wav', 'peak_4.wav', 'peak_5.wav', 'peak_6.wav', 'peak_7.wav', 'peak_8.wav', ]
    measured_LAFmax = [89.9, 80.2, 91.7, 87, 90.1, 90.5, 84.1, 89.6]

    adjusts = []  # the offset value from FFT to LAeq,T
    for v, fn in enumerate(filenames):
        samplerate, data_slice = wavfile.read(os.path.join('validate adjust level', fn))
        plt.plot(data_slice)
        plt.show()
        # no window is applied as the signal lenght is too short. After testing
        # after testing, the uncertainty without window is the least fluctuated ones
        xf, levels = fft_wave_data(samplerate, data_slice, NFFT=16384*2)
        total_a = FFT_to_LAeq(xf, levels)
        adjusts.append(total_a - measured_LAFmax[v])
    adjust = np.mean(adjusts)  # return 43.07 in this case 
    return adjust
        

def FFT_to_LAeq(xf, FFT_levels):
    fft_df = pd.DataFrame({'Frequency': xf, 'Level_dB': FFT_levels})
    a_w = A_weighting(1) # 31.5 Hz to 16k Hz
    flower, fcentre, fupper = octave_bands() # fcentre from 31.5 Hz to 16k Hz
    level_lin = []

    for m in range(len(fcentre)):
        low, up = flower[m], fupper[m]
        temp_df = fft_df[(fft_df['Frequency'] > low) & (fft_df['Frequency'] <= up)]
        levels = temp_df['Level_dB'].values
        Lp = 10.*np.log10(sum(10**(levels/10)))
        level_lin.append(Lp)

    level_a_not_adjusted = np.array(level_lin) + a_w
    total_a = 10.*np.log10(sum(10**(level_a_not_adjusted/10)))
    return total_a


def calc_LAFmax_dt(filename):
    ''' perform FFT and calculate the LAFmax_dt of the wave file'''
    # calcuated noise level minus this value to get the measured value of LAFmax
    adjust = 43.07   # the is the difference between FFT output and measured noise levels.
    samplerate, data_slice = wavfile.read(filename)
    xf, levels = fft_wave_data(samplerate, data_slice, NFFT=16384*2)  
    LAeq_dt = FFT_to_LAeq(xf, levels)
    LAeq_adjusted = LAeq_dt - adjust
    return LAeq_adjusted



def main():
    os.chdir(r'G:\Shared drives\Apex\Acoustic Data\IOA, conference proceedings\2021 papers\Hockey match maximum noise levels')
    adjust_value_by_LAeqT()
    # LAeq_adjusted = calc_Laeq_dt('110821-110824 max.wav')
    # print(LAeq_adjusted)


if __name__ =='__main__':
    main()