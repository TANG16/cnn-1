% MNIST set loading
imageSize = 28;
addpath('common');
trainData = loadMNISTImages('common\train-images-idx3-ubyte');
trainData = reshape(trainData,imageSize,imageSize,1,[]);
trainLabels = loadMNISTLabels('common\train-labels-idx1-ubyte');
trainLabels(trainLabels==0) = 10; % Remap 0 to 10
testData = loadMNISTImages('common\t10k-images-idx3-ubyte');
testData = reshape(testData,imageSize,imageSize,1,[]);
testLabels = loadMNISTLabels('common\t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%% NET config
insize = [size(trainData,1), size(trainData,2)]; %2-dmensional input for conv, otherwise - 1-dim and reshape
NN.scale = 1;
NN.struct = {insize insize/2 (insize/2)/2 (insize/2)/4 10 (insize/2)/4 (insize/2)/2 insize/2 insize}; 
% NN.struct = {insize 150 100 25 insize}; % network structure
NN.afun = {'linear', 'relu', 'relu', 'relu', 'tanh', 'relu', 'relu', 'relu', 'linear'};
NN.ltype ={'c',		 'p', 	 'c',    'p', 	  'f', 	  'dp',  'dc', 	 'dp', 	 'dc'};
NN.lconf = {[4,5],    2,     [6,5],   2,      inf,    2,     [6,5],   2,     [4,5]};%for conv layer - [numFilters, filterDim], for pooling - poolDim, for fully con - none
NN.type = 'DAE'; %'DNN', 'DCAE'
NN.pretrain =  false; 
% NN.atoms = dirname1(num+2).name;

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

%% Initialization
for i = 2:length(NN.struct)
    if(strcmp(NN.ltype{i},'c') || strcmp(NN.ltype{i},'dc'))		%convolutional/deconvolutional layer
        numFilters2 = layer.numFilters;
        filterDim = layer.filterDim;
        layer.W = 1e-1*randn(filterDim,filterDim,numFilters1,numFilters2);
        layer.b = zeros(numFilters2,1);
        layer.W_velocity = zeros(size(layer.W));
        layer.b_velocity = zeros(size(layer.b));      
        convDim = LastOutDim - layer.filterDim + 1;
        layer.delta = zeros(convDim,convDim,numFilters2,opts.batchsize);
        numFilters1 = numFilters2;
        LastOutDim = convDim;	
	elseif(strcmp(NN.ltype{i},'p') || strcmp(NN.ltype{i},'dp'))	%pooling/un-pooling layer
        pooledDim = LastOutDim / layer.poolDim;
        layer.delta = zeros(pooledDim,pooledDim,numFilters1,opts.batchsize);
        LastOutDim = pooledDim;		
	else 														%fully connected layer
		eps_initt = sqrt(6)/sqrt((NN.struct{i} + NN.struct{i-1}));
		NN.W{i-1} = randn(NN.struct{i}, NN.struct{i-1}) * eps_initt;
		NN.B{i-1} = zeros(NN.struct{i},1);
	end
end



tic
[NN,out] = nnTrain(NN,  X, X, [], [], opts);
toc

%% SAVE net after training
% net_name = (['NN_' dirname1(num+2).name '_' ...
% char(datetime('now', 'Format', 'yyyy-MM-dd_HH-mm'))]);
net_name = (['NN_test1' NN.atoms]);
save(net_name, 'NN');