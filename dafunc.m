%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dafunc.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ y ] = dafunc( z_in, z_out, afun)

switch afun
    case 'tanh'                 % tanh units
    y = 1 - z_out.^2;
    case 'linear'               % linear units
    y = 1;
    case 'sigmoid'              % sigmoid units
    y = z_out .* (1 - z_out);
    case 'staircase'            % staircase units
    y = cattanderiv(z_in);
    case 'relu'                 % linear rectified units max(0,x)
    y = double(z_out > 0);       
    case 'quadratic'            % quadratic units
    y = 2.*z_in.*sign(z_in);
    otherwise                   % sigmoid units by default
    y = z_out .* (1 - z_out);
end
        
end