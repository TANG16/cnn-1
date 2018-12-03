% MNIST set loading
imageSize = 28;
addpath('common');
images = loadMNISTImages('common\train-images-idx3-ubyte');
images = reshape(images,imageSize,imageSize,1,[]);
labels = loadMNISTLabels('common\train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10
testImages = loadMNISTImages('common\t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageSize,imageSize,1,[]);
testLabels = loadMNISTLabels('common\t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10