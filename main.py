from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
import numpy as np
from librosa.feature import mfcc
from tqdm import tqdm
from sklearn import *
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from keras.models import *
from keras.layers import *
from keras.layers.convolutional import *
from keras.callbacks import *
import librosa
from os import listdir
from os.path import isfile, join, splitext
import os.path
import librosa.display
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder



# Bu yöntem, gruplandırma ve veri türüne göre verileri alır.
def get_data(clips, grouping="default", dtype="clip"):
    if grouping == "default":
        return get_data_default(clips, dtype)

    return None  # Giriş geçersizse, hiçbir şey döndürme


# Bu yöntem, verileri 4 sınıfa böler -> ["Normal","Wheeze","Crackle","Both"]
def get_data_default(clips, dtype):
    data = [[], [], [], []]

    # Her klipte dolaşın ve klipleri hırıltı/çıtırtı algılamaya göre gruplayın
    for clip in tqdm(clips, "Grouping data by default"):
        c = clip.crackle
        w = clip.wheeze
        # Klibi ilgili listeye ekle
        i = get_index(c, w)
        if dtype == "audio":
            data[i].append(clip.audio)
        elif dtype == "clip":
            data[i].append(clip)
        else:
            print("Uygun olmayan tip bulundu:", dtype)
            print("dtype için uygun değerler şunlardır:", ["audio", "clip"])

    return data  # Gruplandırılmış verileri döndür


# Bu yöntem, verileri kayıt ekipmanına göre gruplanmış 4 sınıfa böler.
def get_data_recording(clips, dtype):
    # Column 0: normal
    # Column 1: crackle
    # Column 2: wheeze
    # Column 3: both
    akgc417l = [[], [], [], []]
    littc2se = [[], [], [], []]
    litt3200 = [[], [], [], []]
    meditron = [[], [], [], []]

    # Her klipte dolaşın ve gruplandırın
    for clip in tqdm(clips, "Grouping data by recording equipment"):

        # Klipten bilgi alın
        c = clip.crackle
        w = clip.wheeze
        r = clip.rec_equipment

        # Booolean hırıltı/çıtırtı bilgisine dayalı indeks atama
        i = get_index(c, w)

        # Klibi türe göre ilgili listeye ekle
        if dtype == "audio":
            if r == "AKGC417L":
                akgc417l[i].append(clip.audio)
            elif r == "LittC2SE":
                littc2se[i].append(clip.audio)
            elif r == "Litt3200":
                litt3200[i].append(clip.audio)
            elif r == "Meditron":
                meditron[i].append(clip.audio)
            else:
                print("Uygunsuz kayıt ekipmanı bulundu:", r)
        elif dtype == "clip":
            if r == "AKGC417L":
                akgc417l[i].append(clip)
            elif r == "LittC2SE":
                littc2se[i].append(clip)
            elif r == "Litt3200":
                litt3200[i].append(clip)
            elif r == "Meditron":
                meditron[i].append(clip)
            else:
                print("Uygunsuz kayıt ekipmanı bulundu:", r)
        else:
            print("Uygun olmayan tip bulundu:", dtype, "->", clip.file_name)
            print("dtype için uygun değerler şunlardır:", ["audio", "clip"])

    return akgc417l, littc2se, litt3200, meditron


def split_data(classes, train=0, test=0, valid=0):
    # Bölmelerin doğru olup olmadığını kontrol edin
    if test + valid + train != 1:
        print("Bölmeler %100'e ulaşmıyor")
        return None
    # Test, geçerli ve tren setleri için dizi uzunlukları atama
    splits = np.zeros((3, len(classes)), dtype=int)
    i = 0
    for c in classes:
        total = len(c)
        splits[0, i] = int(train * total)
        splits[1, i] = int(test * total) + splits[0, i]
        splits[2, i] = total
        i += 1
    # Veri kümesini böl
    train_split = []
    test_split = []
    valid_split = []
    for n in range(len(classes)):
        train_split.append(classes[n][0:splits[0, n]])
        test_split.append(classes[n][splits[0, n]:splits[1, n]])
        valid_split.append(classes[n][splits[1, n]:splits[2, n]])

    return train_split, test_split, valid_split


def crop_clips(clips, seconds, sr):
    n_samples = int(seconds * sr)
    for clip in tqdm(clips, "Cropping clips"):
        audio = clip.sound_data
        if len(audio) < n_samples:
            clip.cropped_sound = np.pad(audio, (0, n_samples - len(audio)))
        elif len(audio) > n_samples:
            clip.cropped_sound = audio[:n_samples]
        else:
            clip.cropped_sound = audio

    return clips


def filter_clips(clips, lower, upper, sr):
    n_lower = lower * sr
    n_upper = upper * sr
    output = []
    for clip in tqdm(clips, "Filtering clips"):
        l = len(clip.sound_data)
        if l >= n_lower and l <= n_upper:
            output.append(clip)

    return crop_clips(output, lower)


def get_index(c, w):
    if c and w:
        return 3
    elif not c and not w:
        return 0
    elif c:
        return 1
    else:
        return 2


class Recording:
    def __init__(self, filename=None, sr=None, data=None):
        self.filename = splitext(filename)[0] + '.wav' if filename else None
        self._sr = sr
        self._data = data

    @property
    def data(self):
        if self._data is None:
            if self.filename is None:
                return None

            self._data, self._sr = librosa.load(self.filename, sr=self.sr,
                                                mono=False, dtype=np.float32)
        return self._data, self._sr

    @property
    def sr(self):
        return self._sr

    def __str__(self):
        return splitext(os.path.split(self.filename)[1])[0]


class Clip:
    mfcc = None
    cropped_sound = None

    def __init__(self, recording, patient_id, rec_i, chest_loc, acq_mode,
                 rec_equipment, crackle, wheeze, start_t=None, end_t=None):

        self.recording = recording
        self.patient_id = patient_id if isinstance(patient_id, int) else int(patient_id)
        self.rec_i = rec_i
        self.chest_loc = chest_loc
        self.acq_mode = acq_mode
        self.rec_equipment = rec_equipment
        self.crackle = crackle
        self.wheeze = wheeze
        self.start_t = start_t
        self.end_t = end_t

    def __str__(self):
        return "Clip({}_{}_{}_{}_{}, c={}, w={}, ({:0.3f}s, {:0.3f}s))".format(self.patient_id, self.rec_i,
                                                                               self.chest_loc, self.acq_mode,
                                                                               self.rec_equipment, int(self.crackle),
                                                                               int(self.wheeze), self.start_t,
                                                                               self.end_t)

    def __repr__(self):
        return self.__str__()

    @property
    def sr(self):
        sr = self.recording.sr if self.recording else None
        return sr

    def recording_name(self):
        return str(self.recording)

    @property
    def sound_data(self):
        (sound_data, sr) = self.recording.data
        if self.start_t is None:
            self.start_t = 0
        if self.end_t is None:
            self.end_t = librosa.samples_to_time(sound_data.size, sr)

        start_i = librosa.time_to_samples(self.start_t, sr)
        end_i = librosa.time_to_samples(self.end_t, sr)

        return sound_data[start_i:end_i]

    @staticmethod
    def parse_annotations(filename):
        # Bir tane olması durumunda uzantıyı kaldırın
        filename = splitext(filename)[0]
        annotations = []
        with open("{}.txt".format(filename)) as f:
            for line in f:
                annotations.append([float(num) if i < 2 else bool(int(num))
                                    for i, num in enumerate(line.split())])

        return annotations

    @classmethod
    def generate_from_file(cls, filename, sr=None, lazy=False):
        # Bir tane olması durumunda uzantıyı kaldırın
        filename_ = splitext(filename)[0]
        if not lazy:
            (sound_data, sr) = librosa.load("{}.wav".format(filename_),
                                            sr=sr, mono=False,
                                            dtype=np.float32)
            recording = Recording(filename, sr, sound_data)
        else:
            recording = Recording(filename, sr)

        annotations = cls.parse_annotations("{}".format(filename_))
        metadata = tuple(os.path.split(filename_)[-1].split("_"))

        clips = []
        for a in annotations:
            clips.append(cls(recording, patient_id=metadata[0],
                             rec_i=metadata[1], chest_loc=metadata[2],
                             acq_mode=metadata[3], rec_equipment=metadata[4],
                             crackle=a[2], wheeze=a[3],
                             start_t=a[0], end_t=a[1]))
        return clips


def import_all_files(directory, sr=None, lazy=False):
    filenames = set(splitext(f)[0].split()[0] for f in listdir(directory)
                    if isfile(join(directory, f)))

    # Bu, klip listelerinin bir listesini üretir
    clips = [Clip.generate_from_file(join(directory, f), sr=sr, lazy=lazy)
             for f in tqdm(filenames, "Files to Clips")]

    clips = [item for sublist in clips for item in sublist]

    return clips


if __name__ == "__main__":
    directory = "C:\\Users\\" ############ Bu kısma dataset veriyolu girilmelidir.
    clips = import_all_files(directory, lazy=True)

    print(set(clip.rec_equipment for clip in clips))  # Kayıt cihazlarının listesini yazdır
directory = "C:\\Users\\" ############ Bu kısma dataset veriyolu girilmelidir.


sample_rate = 44100
clips = import_all_files(directory, sample_rate)

# %% Check sample rates
sample_rates = {}
for clip in clips:
    key = str(clip.sr)
    if key in sample_rates:
        sample_rates[key] += 1
    else:
        sample_rates[key] = 1

print("Sample Rates:")
print(sample_rates)
print()

# %% Analyze clip lengths
times = []
for clip in clips:
    sr = clip.sr
    n = len(clip.sound_data)
    t = n / sr

    times.append(t)
times = np.array(times)

# Randomly select 10 clips and print out their lengths
key = np.random.randint(0, len(clips), 10)
for k in key:
    print(clips[k].recording, times[k])

# %%
plt.figure()
plt.plot(times, '.')
plt.xlabel("Clip Number")
plt.ylabel("Time in Seconds")
plt.title("Plot of Clip Lengths")
plt.savefig('save.png')
plt.show()

# %% Get clips (all at same sample rate for ease of use)
sr = 44100

print("started")

clips = import_all_files(directory, sr)

# %% Crop/filter clips
clips = crop_clips(clips, 5, sr)
# clips = filter_clips(clips,5,6,sr)

# %% Do mfcc on cropped audio
for clip in tqdm(clips, "Doing MFCC"):
    clip.mfcc = mfcc(y=clip.cropped_sound, sr=sr)
    clip.flattened_mfcc = clip.mfcc.flatten()

# %% Separate data by class
data = get_data(clips, grouping="default", dtype="clip")

# %% Split data into training, testing, and validation sets/labels
for d in data:
    random.shuffle(d)
train_split, test_split, valid_split = split_data(data, train=0.8, test=0.2, valid=0.0)

train_data = []
test_data = []
valid_data = []
train_labels = []
test_labels = []
valid_labels = []

i = 0
for clips in tqdm(train_split, "Training split"):
    for clip in clips:
        train_data.append(clip.flattened_mfcc)
        train_labels.append(i)
    i += 1

i = 0
for clips in tqdm(test_split, "Testing split"):
    for clip in clips:
        test_data.append(clip.flattened_mfcc)
        test_labels.append(i)
    i += 1
i = 0
for clips in tqdm(valid_split, "Validation split"):
    for clip in clips:
        valid_data.append(clip.flattened_mfcc)
        valid_labels.append(i)
    i += 1

# %% Scaling the input to standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# %% Applying PCA to reduce dimension while still keeping 99% of the original variance
pca = decomposition.PCA(n_components=0.99, svd_solver='full')
pca.fit(train_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)

params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                'C': [1, 10, 100, 1000]},
               {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
encoder =LabelEncoder()
encoder.fit(train_labels)
Y_train = encoder.transform(train_labels)
# svm_model = GridSearchCV(SVC(probability=False), params_grid, cv=5, n_jobs=-1, refit = True, verbose = 3) # n_jobs=-1 makes sure you use all available cores
svm_model = RandomizedSearchCV(SVC(probability=False), params_grid, n_iter=10, cv=5, n_jobs=-1, refit=True, verbose=10)
print("SVM started")
svm_model.fit(train_data, Y_train)
print("SVM finished")

# %% Choosing the best model and testing
final_model = svm_model.best_estimator_
Y_pred = final_model.predict(test_data)
# print(Y_pred)
Y_pred_label = list(encoder.inverse_transform(Y_pred))
# print(Y_pred_label)
print(confusion_matrix(test_labels, Y_pred_label))
print("\n")
print(classification_report(test_labels, Y_pred_label))

# Get clips
clips = import_all_files(directory)

# Get data and test separated only by class
data0 = get_data(clips, grouping="default", dtype="clip")

# Test if all clips in first group are normal (No wheezes or crackles)
for clip in data0[0]:
    if clip.crackle or clip.wheeze:
        print("Test failed, line 24", clip.crackle, clip.wheeze)
        break
for clip in data0[1]:
    if not (clip.crackle and not clip.wheeze):
        print("Test failed, line 28", clip.crackle, clip.wheeze)
        break
for clip in data0[2]:
    if not (not clip.crackle and clip.wheeze):
        print("Test failed, line 32", clip.crackle, clip.wheeze)
        break
for clip in data0[3]:
    if not (clip.crackle and clip.wheeze):
        print("Test failed, line 36", clip.crackle, clip.wheeze)
        break

# Get data and test separated by class and recording equipment
#r0, r1, r2, r3 = get_data(clips, grouping="recording equipment", dtype="clip")

# Get grouping information
total_clips = len(clips)

total_normal = len(data0[0])
total_crackle = len(data0[1])
total_wheeze = len(data0[2])
total_both = len(data0[3])
total_data0 = total_normal + total_crackle + total_wheeze + total_both

# count_r = np.array([[len(r0[0]), len(r0[1]), len(r0[2]), len(r0[3])],
#                     [len(r1[0]), len(r1[1]), len(r1[2]), len(r1[3])],
#                     [len(r2[0]), len(r2[1]), len(r2[2]), len(r2[3])],
#                     [len(r3[0]), len(r3[1]), len(r3[2]), len(r3[3])]])
# count_r_T = count_r.transpose()

# Test if groupings are accurate
print()
print()
print("Total number of clips:   ", len(clips))
print()
print("Number of normal clips:  ", total_normal)
print("Number of crackle clips: ", total_crackle)
print("Number of wheeze clips:  ", total_wheeze)
print("Number of both clips:    ", total_both)
print("Total check:             ", total_clips == total_data0)
print()
# print("Number of AKGC417L: ", np.sum(count_r[0]))
# print("Number of LittC2SE: ", np.sum(count_r[1]))
# print("Number of Litt3200: ", np.sum(count_r[2]))
# print("Number of Meditron: ", np.sum(count_r[3]))
# print("Class check - norm: ", np.sum(count_r_T[0]) == total_normal)
# print("Class check - crac: ", np.sum(count_r_T[1]) == total_crackle)
# print("Class check - whee: ", np.sum(count_r_T[2]) == total_wheeze)
# print("Class check - both: ", np.sum(count_r_T[3]) == total_both)
# print("Total check:        ", total_clips == np.sum(count_r))

# %%
# clips = import_all_files(directory)

# %% Get data and test separated only by class
data = get_data(clips, grouping="default", dtype="clip")

# %% Do mfcc on every clip
c = 1
images = [[], [], [], []]
for group in data:
    for clip in tqdm(group, "Taking MFCC of clips in group " + str(c) + " of " + str(len(data))):
        clip.mfcc = mfcc(y=clip.sound_data, sr=clip.sr)
    c += 1

# %% Plot random mfccs from each group
c = 0
# for group in data:
#     # Get images to plot
#     key = np.random.randint(0, len(group),size=4)
#     arr = []
#     for n in range(4):
#         arr.append(group[key[n]])
#
#     plt.figure(dpi=500)
#
#     for n in range(4):
#         clip = group[key[n]]
#         plt.subplot(2, 2, n + 1)
#         librosa.display.specshow(clip.mfcc, x_axis='time', y_axis='mel', sr=clip.sr)
#         plt.title(clip.recording)
#
#     plt.tight_layout(pad=3.0)
#
#     plt.show()

print("started")

# clips = import_all_files(directory, sr)

# %% Crop/filter clips
# clips = crop_clips(clips, 5, sr)
# clips = filter_clips(clips,5,6,sr)

# %% Do mfcc on cropped audio
for clip in tqdm(clips, "Doing MFCC"):
    clip.mfcc = mfcc(y=clip.cropped_sound, sr=sr)
    clip.flattened_mfcc = clip.mfcc.flatten()

# %% Separate data by class
data = get_data(clips, grouping="default", dtype="clip")

# %% Split data into training, testing, and validation sets/labels
for d in data:
    random.shuffle(d)
train_split, test_split, valid_split = split_data(data, train=0.7, test=0.2, valid=0.1)

train_data = []
test_data = []
valid_data = []
train_labels = []
test_labels = []
valid_labels = []

i = 0
for clips in tqdm(train_split, "Training split"):
    for clip in clips:
        train_data.append(clip.flattened_mfcc)
        train_labels.append(i)
    i += 1

i = 0
for clips in tqdm(test_split, "Testing split"):
    for clip in clips:
        test_data.append(clip.flattened_mfcc)
        test_labels.append(i)
    i += 1
i = 0
for clips in tqdm(valid_split, "Validation split"):
    for clip in clips:
        valid_data.append(clip.flattened_mfcc)
        valid_labels.append(i)
    i += 1

# %% Scaling the input to standardize features by removing the mean and scaling to unit variance

train_data = scaler.transform(train_data)
valid_data = scaler.transform(valid_data)
test_data = scaler.transform(test_data)

# %% Applying PCA to reduce dimension while still keeping 99% of the original variance
# pca = decomposition.PCA(n_components=0.99, svd_solver = 'full')
# pca.fit(train_data)
# train_data = pca.transform(train_data)
# valid_data = pca.transform(valid_data)
# test_data = pca.transform(test_data)

# %% Plots for visualization
# # Plotting number of component vs explained variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

# # Scatter plot of the classes with most informative 3 principle components
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], c=train_labels[:,0])

# %% One-hot encoding of the labels
train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)

train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
valid_data = valid_data.reshape(valid_data.shape[0], valid_data.shape[1], 1)
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, validX, validy):
    # history = History()
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1)
    verbose, epochs, batch_size = 10, 50, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    # fit network
    # hist=model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, callbacks=[history], validation_split = 0.1)
    hist = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, callbacks=[es],
                     validation_data=(validX, validy))
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=10)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
# def run_experiment(trainX, trainy, testX, testy, validX, validy, repeats=10):
#     # repeat experiment
#     scores = list()
#     for r in range(repeats):
#         score = evaluate_model(trainX, trainy, testX, testy, validX, validy)
#         score = score * 100.0
#         print('>#%d: %.3f' % (r + 1, score))
#         scores.append(score)
#     # summarize results
#     return summarize_results(scores)
#
#
# # run the experiment
# run_experiment(train_data, train_labels, test_data, test_labels, valid_data, valid_labels, repeats=10)

# %% Get clips (all at same sample rate for ease of use)
# sr = 44100

print("started")

# clips = import_all_files(directory, sr)

# %% Crop/filter clips
# clips = crop_clips(clips, 5, sr)
# clips = filter_clips(clips,5,6,sr)

# %% Do mfcc on cropped audio
for clip in tqdm(clips, "Doing MFCC"):
    clip.mfcc = mfcc(y=clip.cropped_sound, sr=sr)
    clip.flattened_mfcc = clip.mfcc.flatten()

# %% Separate data by class
data = get_data(clips, grouping="default", dtype="clip")

# %% Split data into training, testing, and validation sets/labels
for d in data:
    random.shuffle(d)
train_split, test_split, valid_split = split_data(data, train=0.7, test=0.2, valid=0.1)

train_data = []
test_data = []
valid_data = []
train_labels = []
test_labels = []
valid_labels = []

i = 0
for clips in tqdm(train_split, "Training split"):
    for clip in clips:
        # train_data.append(clip.flattened_mfcc)
        train_data.append(clip.mfcc)
        train_labels.append(i)
    i += 1

i = 0
for clips in tqdm(test_split, "Testing split"):
    for clip in clips:
        # test_data.append(clip.flattened_mfcc)
        test_data.append(clip.mfcc)
        test_labels.append(i)
    i += 1
i = 0
for clips in tqdm(valid_split, "Validation split"):
    for clip in clips:
        # valid_data.append(clip.flattened_mfcc)
        valid_data.append(clip.mfcc)
        valid_labels.append(i)
    i += 1

# %% Convert list to 3D array
train_data = np.asarray(train_data)
valid_data = np.asarray(valid_data)
test_data = np.asarray(test_data)

# %% Scaling the input to standardize features by removing the mean and scaling to unit variance
# scaler = preprocessing.StandardScaler().fit(train_data)
# train_data = scaler.transform(train_data)
# valid_data=scaler.transform(valid_data)
# test_data=scaler.transform(test_data)

# %% Applying PCA to reduce dimension while still keeping 99% of the original variance
# pca = decomposition.PCA(n_components=0.99, svd_solver = 'full')
# pca.fit(train_data)
# train_data = pca.transform(train_data)
# valid_data = pca.transform(valid_data)
# test_data = pca.transform(test_data)

# %% Plots for visualization
# # Plotting number of component vs explained variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

# # Scatter plot of the classes with most informative 3 principle components
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], c=train_labels[:,0])

# %% One-hot encoding of the labels
train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)

train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
valid_data = valid_data.reshape(valid_data.shape[0], valid_data.shape[1], valid_data.shape[2], 1)
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)


def evaluate_2d_cnn_model(train_data, train_labels, test_data, test_labels, valid_data, valid_labels):
    ### Building the model
    hidden_num_units = 2048
    hidden_num_units1 = 1024
    hidden_num_units2 = 128
    output_num_units = train_labels.shape[1]

    epochs = 50  # 10
    batch_size = 16  # 16
    pool_size = (2, 2)
    input_shape = Input(shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3]))
    kernel_size = (3, 3)

    model = Sequential([

        Conv2D(16, kernel_size, activation='relu',
               input_shape=(train_data.shape[1], train_data.shape[2], train_data.shape[3]), padding='same'),
        BatchNormalization(),

        Conv2D(16, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),

        Conv2D(32, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),

        Conv2D(32, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),

        Conv2D(64, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),

        Conv2D(64, kernel_size, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool_size),
        Dropout(0.2),

        Flatten(),

        Dense(units=hidden_num_units, activation='relu'),
        Dropout(0.3),
        Dense(units=hidden_num_units1, activation='relu'),
        Dropout(0.3),
        Dense(units=hidden_num_units2, activation='relu'),
        Dropout(0.3),
        Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'),
    ])

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model is compiled")
    model.summary()

    ### Training the model
    trained_model_conv = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                                   validation_data=(valid_data, valid_labels))
    print("Model is trained")

    ### Prdicting the class
    pred = model.predict_classes(test_data)

    ### Evaluating the model
    scores = model.evaluate(test_data, test_labels)

    print(model.metrics_names)
    print(scores)

    acc = trained_model_conv.history['accuracy']
    val_acc = trained_model_conv.history['val_accuracy']
    loss = trained_model_conv.history['loss']
    val_loss = trained_model_conv.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('read2.png')
    plt.show()
    return scores


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(trainX, trainy, testX, testy, validX, validy, repeats=10):
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_2d_cnn_model(trainX, trainy, testX, testy, validX, validy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


# run the experiment
# run_experiment(train_data, train_labels, test_data, test_labels, valid_data, valid_labels, repeats=1)

for clip in tqdm(clips, "Doing MFCC"):
    clip.mfcc = mfcc(y=clip.cropped_sound, sr=sr)
    clip.flattened_mfcc = clip.mfcc.flatten()

# %% Separate data by class
data = get_data(clips, grouping="default", dtype="clip")

# %% Split data into training, testing, and validation sets/labels
for d in data:
    random.shuffle(d)
train_split, test_split, valid_split = split_data(data, train=0.6, test=0.2, valid=0.2)

train_data = []
test_data = []
valid_data = []
train_labels = []
test_labels = []
valid_labels = []

for clips in tqdm(train_split, "Training split"):
    i = 0
    for clip in clips:
        train_data.append(clip.flattened_mfcc)
        train_labels.append(i)
        i += 1

for clips in tqdm(test_split, "Testing split"):
    i = 0
    for clip in clips:
        test_data.append(clip.flattened_mfcc)
        test_labels.append(i)
        i += 1

for clips in tqdm(valid_split, "Validation split"):
    i = 0
    for clip in clips:
        valid_data.append(clip.flattened_mfcc)
        valid_labels.append(i)
        i += 1

# # %% Applying PCA
# pca = decomposition.PCA(n_components=0.2, svd_solver='full')
# pca.fit(train_data)
# data_matrix_new = pca.transform(train_data)
#
# # Plotting number of component vs explained variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()





