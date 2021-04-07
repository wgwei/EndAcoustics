from matplotlib.cm import ScalarMappable
from numpy.core.fromnumeric import mean
from numpy.testing._private.utils import measure
from scipy.io import wavfile
import os
import matplotlib.pylab as plt
from scipy.fft import fft, fftfreq
import numpy as np
import pandas as pd
from scipy.signal.windows import hann

os.chdir(r'G:\Shared drives\5000 - 5999\_5800 - 5899\5805 Harrogate Football Club\Calculations\201214 Visit 3 Survey\P2 - Further down St Nicholas Road\2020-12-14_SLM_002\WW edit for LA90')

def split_data():
    samplerate, data = wavfile.read('MS_PCM_signed_16bit.wav')
    print('Sampling rate ->', samplerate)
    print('Number of samples ->', data.shape)

    how_many_minutes  = int(data.shape[0] / samplerate / 60)
    for n in range(how_many_minutes):
        if n==0:
            wavfile.write('PCM_'+str(n)+'_minute.wav', samplerate, data[0:28*samplerate])    # the first 28 seconds   
        else:
            wavfile.write('PCM_'+str(n)+'_minute.wav', samplerate, data[n*60*samplerate:(n+1)*60*samplerate])
    print('done')


def fft_wave(filename):
    ''' file format should be PCM or IEEE format. if not, using such as Audacity to convert to PCM.
    '''
    samplerate, data_1  = wavfile.read(filename)
    NFFT = 16384 * 2
    w = hann(data_1.shape[0])
    yf = fft(data_1*w,  NFFT)
    xf = fftfreq(NFFT, 1/NFFT)[0:NFFT//2] * samplerate/NFFT
    yf_amplitude = 2./NFFT*np.abs(yf[0:NFFT//2])

    p0 = 0.00002 
    levels = 20*np.log10(yf_amplitude/p0)
    return xf, levels


def octave_bands():
    c = np.array(range(-5, 5, 1))
    fcentre  = 1000 * (2.**c)
    fd = 2.0**(1/2)
    fupper = fcentre * fd
    flower = fcentre / fd
    return flower, fcentre, fupper


def validate_with_octave():
    filenames = ['PCM_0_minute.wav', 'PCM_1_minute.wav', 'PCM_2_minute.wav', 'PCM_3_minute.wav']
    Laeq = pd.read_csv('2020-12-14_SLM_002_csv.csv')
    measured_octave_levels = Laeq.iloc[0:4, 6::]

    for n in range(len(filenames)):
        plt.subplot(2, 2, n+1)
        xf, levels = fft_wave(filenames[n])
        fft_df = pd.DataFrame({'Frequency': xf, 'Level_dB': levels})
        A_weighting = np.array([-39.4, -26.2, -16.1, -8.6, -3.2, 0, 1.2, 1, -1.1, -6.6]) # 31.5 Hz to 16k Hz
        flower, fcentre, fupper = octave_bands() # fcentre from 31.5 Hz to 16k Hz
        level_lin = []

        for m in range(len(fcentre)):
            low, fc, up = flower[m], fcentre[m], fupper[m]
            temp_df = fft_df[(fft_df['Frequency'] > low) & (fft_df['Frequency'] <= up)]
            levels = temp_df['Level_dB'].values
            Lp = 10.*np.log10(sum(10**(levels/10)))
            level_lin.append(Lp)

        level_a_not_adjusted = np.array(level_lin) + A_weighting
        total_a = 10.*np.log10(sum(10**(level_a_not_adjusted/10)))
        v = measured_octave_levels.iloc[n, :].values
        dBA = 10*np.log10(sum(10**(v/10)))
        adjust = total_a - dBA
        print('Adjust: ', adjust)
        level_a = level_a_not_adjusted - adjust

        plt.loglog(fcentre[0:8], level_a[0:8])
        plt.loglog(fcentre[0:8], measured_octave_levels.iloc[n, 0:8])
        plt.legend(['Calc', 'Measured'])
        plt.grid()
    plt.show()


def third_octave_bands():
    c = np.array(range(-15, 10, 1))
    fcentre  = 1000 * (2.0**(c/3))
    fd = 2**(1/6)
    fupper = fcentre * fd
    flower = fcentre / fd
    return flower, fcentre, fupper


def fft_wave_data(samplerate, data, NFFT=16384*2):
    w = hann(data.shape[0])
    yf = fft(data*w,  NFFT)
    xf = fftfreq(NFFT, 1/NFFT)[0:NFFT//2] * samplerate/NFFT
    yf_amplitude = 2./NFFT*np.abs(yf[0:NFFT//2])

    p0 = 0.00002 
    levels = 20*np.log10(yf_amplitude/p0)

    return xf, levels


def get_LA90():
    #a weigting from 31.5 to 8k Hz in 1/3 octave
    A_weighting = np.array([-39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6, 1, 1.2, 1.3, 1.2, 1, 0.5, -0.1, -1.1])
    flower, fcentre, fupper = third_octave_bands() # fcentre from 31.5 Hz to 16k Hz

    samplerate, data = wavfile.read('MS_PCM_signed_16bit.wav')

    how_many_seconds  = int(data.shape[0] / samplerate)
    level_a_each_sec = []
    adjust_list = []
    LAeq1_4s = [47.3, 48.1, 48.9, 49.4]

    # caluclate adjust values
    for n in range(4):
        data_slice = data[n*samplerate:(n+1)*samplerate]
        xf, levels = fft_wave_data(samplerate, data_slice, NFFT=16384*2)
        fft_df = pd.DataFrame({'Frequency':xf, 'Level_dB':levels})

        level_lin = []
        for m in range(len(fcentre)):
            low, fc, up = flower[m], fcentre[m], fupper[m]
            temp_df = fft_df[(fft_df['Frequency'] >= low) & (fft_df['Frequency'] < up)]
            levels = temp_df['Level_dB'].values
            Lp = 10.*np.log10(sum(10**(levels/10)))
            level_lin.append(Lp)

        level_a_not_adjusted = np.array(level_lin) + A_weighting
        total_a = 10.*np.log10(sum(10**(level_a_not_adjusted/10)))
        adjust_list.append(total_a - LAeq1_4s[n])
    adjust = mean(adjust_list)

    # calculuate the spectrum for each sec
    for n in range(how_many_seconds):
        # if n >100:
        #     break
        data_slice = data[n*samplerate:(n+1)*samplerate]
        xf, levels = fft_wave_data(samplerate, data_slice, NFFT=16384*2)
        fft_df = pd.DataFrame({'Frequency':xf, 'Level_dB':levels})

        level_lin = []
        for m in range(len(fcentre)):
            low, fc, up = flower[m], fcentre[m], fupper[m]
            temp_df = fft_df[(fft_df['Frequency'] >= low) & (fft_df['Frequency'] < up)]
            levels = temp_df['Level_dB'].values
            Lp = 10.*np.log10(sum(10**(levels/10)))
            level_lin.append(Lp)

        level_a = np.array(level_lin) + A_weighting - adjust
        level_a_each_sec.append(level_a)
    
    # write to pandas DataFrame and export to csv
    freqs = ['%d' %int(c) for c in fcentre]
    df = pd.DataFrame(np.array(level_a_each_sec), columns=freqs)
    df.to_csv('LAeqT_per_sec.csv') # taking  long time to write file !!
    print(df.tail())

    La90 = []
    for f in freqs:
        ss = df[f].sort_values() # ascending
        La90.append(ss[int(0.1*ss.shape[0])])
    La90_df = pd.DataFrame({'Frequency': freqs, 'LA90_dB': La90})
    La90_df.to_csv('La90_at_tird_octave.csv')
    print(La90_df)


def main():
    # validate_with_octave()
    # split_data()
    # fft_wave('PCM_1_minute.wav')
    get_LA90()


if __name__=='__main__':
    main()





