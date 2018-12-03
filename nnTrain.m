%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nnTrain.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ FF,a] = nnTrain(FF,  X, Y, X_test, Y_test, opts)
m = opts.batchsize;
N =size(X,2);
numbatches = floor(N/opts.batchsize);
err_hist = [];
len = length(FF.W)+1;

for d = 1:len-1 
    dW{d} = zeros(size(FF.W{d})); 
    hdW{d} = zeros(size(FF.W{d}));
    acchdW{d} = zeros(size(FF.W{d}));
    gradW{d} = zeros(size(FF.W{d})); 
    dB{d} = zeros(size(FF.B{d})); 
    hdB{d} = zeros(size(FF.B{d})); 
    gradB{d} = zeros(size(FF.B{d})); 
    acchdB{d} = zeros(size(FF.B{d}));
end
epsi = 1e-6;
err_test = 0;
for epoch = 1:opts.maxepoch 
    randomorder=randperm(N);
    err = 0;
    simerr = 0;
    accuracy = 0;
    err_rel = [];
    for d = 1:len-1
        % Weight save
        W{d} = FF.W{d};
        % Bias unit save
        B{d} = FF.B{d};
    end
    
    
% for it = 1:N
%     [mIdx] = knnsearch(X',X','K',10);
% end
    
for batch = 1:numbatches     
data = X(:,randomorder(1+(batch-1)*opts.batchsize:batch*opts.batchsize));
target = Y(:,randomorder(1+(batch-1)*opts.batchsize:batch*opts.batchsize));

    if opts.denoise>0
      ndata = data+opts.denoise*bsxfun(@times,std(data(:)),randn(size(data)));
%        ndata = data.*binornd(1,opts.denoise,size(data));  
    else
    ndata = data;
    end 
    
%     target = ndata;
    
    [a, z, drop, qnt] = ffprop(FF, ndata, opts.quantize,opts.dropout);
    similar = [];    val = [];
      
    err = err + mse(a{end},target);  
    err_rel = abs((a{end}-target))./abs(target);
    err_rel(isnan(err_rel))=[];
    err_rel(isinf(err_rel))=[];
    err_rel((err_rel)>std(err_rel(:)))=std(err_rel(:));

%     100*sum(sum(1-err_rel))/(size(err_rel,1)*size(err_rel,2))
    accuracy = accuracy + 100*sum(sum(1-err_rel))/(size(err_rel,1)*...
        size(err_rel,2));

    delta = cell(size(a));      
    delta{len} = -(target - a{end}).*dafunc(z{len}, a{len}, FF.afun{len});      
    for d = len-1:-1:2 
        delta{d} = FF.W{d}' * delta{d+1} .* ...
            dafunc(z{d}, a{d}, FF.afun{d}).*drop{d}; 
%         if opts.decorr>0 && d == (len+1)/2
%             tmp = a{d}*a{d}';
%                     delta{d} = (FF.W{d}' * delta{d+1} + (opts.decorr)*( (1/m)*tmp - eye(size(tmp,1)) ) * a{d}) .* ...
%             dafunc(z{d}, a{d}, FF.afun{d}).*drop{d}; 
%         end
    end
    
    for d = 1:len-1
        gradB{d} = sum(delta{d+1},2)/m;
        gradW{d} = (delta{d+1}*a{d}')/m + opts.lambda*FF.W{d};
        if FF.scale > 1
            gradW{d} = gradW{d}.*FF.mask{d};%./FF.scale;
        end
        if opts.quantize && d == (len+1)/2
            gradW{d} = (delta{d+1}*qnt')/m + opts.lambda*FF.W{d};
        end  
    end

    for d = 1:len-1
        switch opts.method
        case 'RMSprop'
        % Weight update
        dW{d} = (1-opts.momentum).*gradW{d}.^2 + opts.momentum.*dW{d};
        FF.W{d} = FF.W{d} - (opts.step./sqrt(dW{d}+eps)).*gradW{d};
        % Bias unit update
        dB{d} = (1-opts.momentum).*gradB{d}.^2 + opts.momentum.*dB{d};
        FF.B{d} = FF.B{d} - (opts.step./sqrt(dB{d}+eps)).*gradB{d};
        case 'adam'
        b1 = 0.9;
        b2 = 0.99;
        hdW{d} = (1-b1).*gradW{d} + b1*hdW{d};
        dW{d} = (1-b2).*gradW{d}.^2 + b2*dW{d};
        FF.W{d} = FF.W{d} - (opts.step.*(hdW{d}/(1-b1^epoch))./...
            sqrt(dW{d}./(1-b2^epoch)+1e-8));
        % Bias unit update
        hdB{d} = (1-b1).*gradB{d} + b1*hdB{d};
        dB{d} = (1-b2).*gradB{d}.^2 + b2.*dB{d};
        FF.B{d} = FF.B{d} - (opts.step.*(hdB{d}/(1-b1^epoch))./...
            sqrt(dB{d}./(1-b2^epoch)+1e-8));
        case 'momentum'
        dW{d} = opts.step*gradW{d} + opts.momentum*dW{d};
        FF.W{d} = FF.W{d} - dW{d};
        % Bias unit update
        dB{d} = opts.step*gradB{d} + opts.momentum*dB{d};
        FF.B{d} = FF.B{d} - dB{d};
        end
    end
end

err_hist = [err_hist err/numbatches];

% Adaptive opts.step size
if strcmp(opts.method,'momentum')
if epoch>1
    if err_hist(end) > 1.01*err_hist(end-1) && opts.step > opts.step_min
        opts.step = opts.step*opts.step_dec;
        % The new weights and biases are discarded
        for d = 1:len-1
            FF.W{d} = W{d};
            FF.B{d} = B{d};
        end
        fprintf('The new weights and biases are DISCARDED!\n');
    elseif err_hist(end) < err_hist(end-1) && opts.step < opts.step_max
        opts.step = opts.step*opts.step_inc;
    else 
        opts.step = opts.step;
    end
end
end

% Validation (if X_test is not empty)
if ~isempty(X_test)
    data = X_test;
    [a, ~,~,~] = ffprop(FF, data, opts.quantize,0);    
    err_test = mse(a{end},Y_test);
end

fprintf('Epoch %d Train error %4.6f Test error %4.6f Accuracy %2.2f %%\n',...
    epoch, 16*err_hist(end), err_test, (accuracy/numbatches));

% Generate output data & Stop conditions
if (epoch == opts.maxepoch) %|| ...
%   (stop_condition(err_hist, opts.eps, opts.err_win_len))
   data = X;
   [a, ~,~,~] = ffprop(FF, data, opts.quantize,0);   
   break;
end
end
end