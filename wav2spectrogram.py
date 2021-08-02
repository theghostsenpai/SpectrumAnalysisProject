from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tempfile import mktemp
import os

def process(p):
    mp3_audio = AudioSegment.from_file(p, format="ogg")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    FS, data = wavfile.read(wname)  # read wav file
    plt.specgram(data, Fs=FS, NFFT=128, noverlap=0)  # plot
    plt.savefig('{}.png'.format(p))
    # plt.show()
    print("{} .... done".format(p))



direc = r'C:\Users\tanmo\Downloads'
def get_file_paths(dirname):
    file_paths = []  
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  
    return file_paths 


files = get_file_paths(direc)
for file in files:
        process(file)

