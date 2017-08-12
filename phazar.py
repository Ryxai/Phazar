import argparse, resampy
import librosa
import numpy as np
from scipy import fft, ifft
from itertools import accumulate

def scale(arr, sample_rate, factor)  :
    return resampy.resample(np.asarray(arr, np.float32), sample_rate,
                            int(sample_rate * factor), axis=-1)

def strech(arr, factor, win_size=1024, hop=None):
    if not hop:
        hop = float(win_size) * 0.25
    window = np.hanning(win_size)
    stft = [fft(window * arr[i:i+win_size]) for i in
                                    range(0, len(arr) - win_size, win_size)]
    if len(stft) < 2:
        raise RuntimeError("Win size too large")
    stft = [(stft[i+1], stft[i]) for i in range(len(stft) - 1)]
    frames = len(stft)
    omegas = [2 * np.pi * np.arange(len(x[0]))/len(x[0]) for x in stft]
    deltas = [np.angle(x[0]) - np.angle(x[1]) for x in stft]
    unwrapped = [deltas[i] - hop * omegas[i] for i in range(frames)]
    rewrapped = [np.mod(x + np.pi, 2 * np.pi) - np.pi for x in unwrapped]
    freqs = [omegas[i] + rewrapped[i]/hop for i in range(frames)]
    phase_acc = list(
        accumulate(freqs, func=lambda acc, x: acc + x * factor * hop))
    cartesians = [np.abs(stft[i][0]) * np.exp(phase_acc[i]*1j)
                                                         for i in range(frames)]
    corrected = [window * np.real(ifft(x)) for x in cartesians]
    x = np.zeros(int(len(corrected) * hop * factor))
    print(len(x))
    print(len(corrected * win_size))
    for i, j in enumerate(range(0, len(x) - win_size, int(factor * hop))):
       # print(len(x[j:j+win_size]))
        #print(len(corrected[i]))
        x[j: j + win_size] += corrected[i]
    return x


def pitch_mod(arr, sample_rate, factor=None, source_Hz=None, out_Hz=None):
    if not factor and (not source_Hz or not out_Hz):
        raise ValueError("Either a pitch modification factor, or i/o hz must "+
                         "be picked")
    if not factor:
        factor = source_Hz/float(out_Hz)
    factor = 2**(factor/12)
    error = 0
    win_size = 1024
    while win_size < 32:
        try:
            arr = strech(arr, factor, win_size=win_size)
            break
        except RuntimeError:
            win_size = win_size/2
    return scale(arr, sample_rate, 1/float(factor))

def time_mod(arr, rate, factor):
    return scale(arr, rate, factor)

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="A phase vocoder written in python")
    parser.add_argument("input", type=str,
                            help="path to an input wav file")
    parser.add_argument("--output", "-o", type=str, default=None,
                            help="path to write output wav")
    parser.add_argument("--pitch","-p", type=float, default=1,
                           help="amount to scale the pitch by > 0")
    parser.add_argument("--speed","-s", type=float, default=1,
                            help="amount to scale the time by")
    args = parser.parse_args()
    wav, rate = librosa.load(args.input)
    wav = librosa.core.to_mono(wav)
    if args.pitch != 1:
        wav = pitch_mod(wav, rate, factor=args.pitch)
    if args.speed != 1:
        wav = time_mod(wav, rate, 1/float(args.speed))
    if not args.output:
        for bit in wav:
            print(bit)
    else:
        librosa.output.write_wav(args.output, wav, rate)


