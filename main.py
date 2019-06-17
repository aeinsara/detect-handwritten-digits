# *-* coding: utf-8 *-*
	
import cv2
import numpy as np

SZ = 32
train_deskew = []
test_deskew = []

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def hog() :
    winSize = (32,32)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog
    #affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def svmInit(C=10, gamma=0.6):
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    return model

def train(model, samples, responses):
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model

def predict(model, samples):
    return model.predict(samples)[1].ravel()

def eval(model, samples, labels):
    predictions = predict(model, samples)
    accuracy = (labels == predictions).mean()
    print('Accuracy = %.2f %%' % (accuracy * 100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

print('################################################################################')
print()

print('Reading Train 60000.cdb ...')
train_images, train_labels = read_hoda_cdb('./DigitDB/Train 60000.cdb')
print("ty", type(train_labels))
print('Reading Test 20000.cdb ...')
test_images, test_labels = read_hoda_cdb('./DigitDB/Test 20000.cdb')

print('Reading RemainingSamples.cdb ...')
remaining_images, remaining_labels = read_hoda_cdb('./DigitDB/RemainingSamples.cdb')

print()

# ******************************************************************************

print('deskew ... ')
train_deskewed = list(map(deskew, train_images))
test_deskewed = list(map(deskew, test_images))

print('HoG ...')
hog = hog()

hog_train = []
hog_test = []

for img in train_deskewed:
    hog_train.append(hog.compute(img))
    #print('eee', hog.compute(img).shape, type(hog.compute(img)) )
for img in test_deskewed:
    hog_test.append(hog.compute(img))

hog_train = np.squeeze(hog_train)
hog_test = np.squeeze(hog_test)

print('eee', np.squeeze(hog_train).shape, type(np.squeeze(hog_train)))

print('Training ...')
model = svmInit()
print(hog_train.shape, type(hog_train))
print(hog_train[0].shape, type(hog_train[0]))
train(model, hog_train, np.asarray(train_labels))


print('Evaluating ... ')
eval(model, hog_test, np.asarray(test_labels))

print('################################################################################')
print()