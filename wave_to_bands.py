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

os.chdir(r'G:\Shared drives\8000 - 8999\8000 - 8099\8021 Crown House, Brighton\Calculations\210616 CJ survey\Crown House Inside BLUE\FFT_audio')

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


def octave_bands():
    c = np.array(range(-5, 5, 1))
    fcentre  = 1000 * (2.**c)
    fd = 2.0**(1/2)
    fupper = fcentre * fd
    flower = fcentre / fd
    return flower, fcentre, fupper


def validate_with_octave():
    filename = 'MS_audio.wav'
    measured_octave_levels = np.array([-9, 2.1, 22.1, 25.4, 14.6, 14.9,  13.6, 12, 9.8, 3.2])  # 31.5 to 16 kHz
    measured_dBA = 27.9

    samplerate, data_slice = wavfile.read(filename)
    xf, levels = fft_wave_data(samplerate, data_slice, NFFT=16384*2)

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
    adjust = total_a - measured_dBA
    print('Adjust: ', adjust)
    level_a = level_a_not_adjusted - adjust

    plt.loglog(fcentre[0:8], level_a[0:8])
    plt.loglog(fcentre[0:8], measured_octave_levels[0:8])
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
    ''' the input are as follows: 
        samplerate, data = wavfile.read(filename)
    '''
    w = hann(data.shape[0]) # Adding hann window may have some effect if the signal is short and impulsive
    yf = fft(data*w,  NFFT)
    # yf = fft(data,  NFFT)
    xf = fftfreq(NFFT, 1/NFFT)[0:NFFT//2] * samplerate/NFFT
    yf_amplitude = 2./NFFT*np.abs(yf[0:NFFT//2])

    p0 = 0.00002 
    levels = 20*np.log10(yf_amplitude/p0)

    return xf, levels


def get_third_octave_levels():
    #a weigting from 31.5 to 8k Hz in 1/3 octave
    A_weighting = np.array([-39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0, 0.6, 1, 1.2, 1.3, 1.2, 1, 0.5, -0.1, -1.1])
    flower, fcentre, fupper = third_octave_bands() # fcentre from 31.5 Hz to 16k Hz

    filename = 'MS_audio.wav'
    samplerate, data_slice = wavfile.read(filename)
    measured_dBA = 27.9

    # caluclate adjust values
    xf, levels = fft_wave_data(samplerate, data_slice, NFFT=16384*2)
    fft_df = pd.DataFrame({'Frequency_Hz':xf, 'Level_dB':levels})
    print(fft_df.head())
    level_lin = []
    for m in range(len(fcentre)):
        low, fc, up = flower[m], fcentre[m], fupper[m]
        temp_df = fft_df[(fft_df['Frequency_Hz'] >= low) & (fft_df['Frequency_Hz'] < up)]
        levels = temp_df['Level_dB'].values
        Lp = 10.*np.log10(sum(10**(levels/10)))
        level_lin.append(Lp)

    level_a_not_adjusted = np.array(level_lin) + A_weighting
    total_a = 10.*np.log10(sum(10**(level_a_not_adjusted/10)))

    adjust = total_a - measured_dBA

    level_a_adjusted = level_a_not_adjusted - adjust
    
    level_a_output = pd.DataFrame({'Frequency_Hz':fcentre, 'Level_dBA':level_a_adjusted})

    fft_df.to_csv('FFT_level_dB.csv')
    level_a_output.to_csv('One_third_octave_level_dBA.csv')

    return level_a_adjusted


def main():
    # validate_with_octave()
    # split_data()
    a = get_third_octave_levels()


if __name__=='__main__':
    main()





