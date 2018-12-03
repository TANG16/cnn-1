%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cattanderiv.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function retval = cattanderiv(theta)
N = 16;
a3 = 200;
j = 1:N-1; 

S = (0.25*a3/((N-1)))*sum(sech(a3*bsxfun(@minus,theta(:)+0.25,0.5*j./N)).^2,2);

retval = reshape(S,size(theta));
end