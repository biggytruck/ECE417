from pickle import dump
import numpy as np


def load_unit_feature(person, word, path='../feature/', num=5):
    """Loader for feature matrices"""
    results = []
    for i in range(1, num+1):
        full_path = '{0}{1}/{1}_{2}{3}.fea'.format(path, person, word, i)
        data = np.genfromtxt(full_path, delimiter=',')
        results.append(data)
    return results


def load_unit_data(person, word, path='../data/', num=5):
    """Loader for wav data"""
    from scipy.io import wavfile
    results = []
    for i in range(1, num+1):
        full_path = '{0}{1}/{1}_{2}{3}.wav'.format(path, person, word, i)
        data = wavfile.read(full_path)
        if len(data[1].shape) == 2: # If has two channels
            data = data[0], data[1][:, 0] # Take the first channel
        results.append(data)
    return results


def load_features(fn):
    """Giving a loader function, generate dict of data"""
    words = "asr cnn dnn hmm tts".split()
    people = "dg ls mh yx".split()
    results_pw = {p: {} for p in people}
    for p in people:
        for w in words:
            results_pw[p][w] = fn(p, w)
    results_wp = {w: {} for w in words}
    for p in people:
        for w in words:
            results_wp[w][p] = fn(p, w)
    return results_pw, results_wp


if __name__ == '__main__':
	print('Saving data features into four pickle files')
	fpw, fwp = load_features(load_unit_feature)  # Features
	dpw, dwp = load_features(load_unit_data)     # Audio
	with open('fea_person_word.pkl', 'wb') as f:
	    dump(fpw, f)
	with open('fea_word_person.pkl', 'wb') as f:
	    dump(fwp, f)
	with open('wav_person_word.pkl', 'wb') as f:
	    dump(dpw, f)
	with open('wav_word_person.pkl', 'wb') as f:
	    dump(dwp, f)
