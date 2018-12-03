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

%% Creation and initialization
insize = size(X,1);
NN.scale = 1;
NN.struct = {insize insize/2 (insize/2)/2 (insize/2)/4 10 (insize/2)/4 (insize/2)/2 insize/2 insize}; 
% NN.struct = {insize 150 100 25 insize}; % network structure
NN.afun = {'linear', 'relu', 'relu', 'relu','tanh', 'relu', 'relu', 'relu', 'linear'};
NN.ltype ={'c',		'p', 	'c',    'p', 	'f', 	'dp', 	'dc', 	'dp', 	'dc'};
NN.type = 'DAE'; %'DNN', 'DCAE'
NN.pretrain =  false; 
NN.atoms = dirname1(num+2).name;

%% Generate random weights (if no pretraining)
for i = 2:length(NN.struct)
    if(strcmp(NN.ltype{i},'c') || strcmp(NN.ltype{i},'dc'))		%convolutional/deconvolutional layer

	
	
	elseif(strcmp(NN.ltype{i},'p') || strcmp(NN.ltype{i},'dp'))	%pooling/un-pooling layer
	
	
	
	else 														%fully connected layer
    eps_initt = sqrt(6)/sqrt((NN.struct{i} + NN.struct{i-1}));
    NN.W{i-1} = randn(NN.struct{i}, NN.struct{i-1}) * eps_initt;
    NN.B{i-1} = zeros(NN.struct{i},1);
	end
end

%% Training setup
opts.method = 'adam';    %'momentum', 'adadelta', 'RMSprop'
opts.step = 0.001;          % learning rate
opts.momentum = 0.9;        % momentum rate
opts.lambda = 1e-4;         % L2-norm scale
opts.maxepoch = 100;         % max number of training iterations
opts.eps = 0.000001;         % min error threshold
opts.batchsize = 200;      % number of samples per batch
opts.quantize = false;      % conditional enabling of quantization
opts.step_dec = 0.7;        % rate of decreasing learning speed
opts.step_inc = 1.05;       % rate of increasing learning speed
opts.step_max = 0.1;        % maximal step
opts.step_min = 0.00001;    % minimal step
opts.err_win_len = 16;      % error control window
opts.denoise = 0.1;
opts.dropout = 0.01;
opts.decorr = 0.0;

tic
[NN,out] = nnTrain(NN,  X, X, [], [], opts);
toc

% net_name = (['NN_' dirname1(num+2).name '_' ...
% char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm'))]);
net_name = (['NN_test1' NN.atoms]);
save(net_name, 'NN');