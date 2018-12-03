%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% afunc.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ y ] = afunc( z, afun)

switch afun
    case 'tanh'         % tanh units
    y = tanh(z);
    case 'linear'       % linear units
    y = z;
    case 'sigmoid'      % sigmoid units
    y = logsig(z);
    case 'staircase'    % staircase units
    y = cattan(z);
    case 'relu'         % linear rectified units
    y = max(0,z);  
    case 'quadratic'    % quadratic units
    y = sign(z).*(z.^2); 
    otherwise           % sigmoid units by default
    y = logsig(z);
end
        
end