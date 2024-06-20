function EHVI = Infill_EHVI(test_x,GPModels,train_y_nds,RefPoint,n_obj) 
% calculate the EHVI at multiple requests, e.g., n for minimization problems

%------------------------------- Notes --------------------------------
% EHVI computation codes are kindly provided by Emmerich et al. [1] and
% Yang et al. [2]. If you are unable to use KMAC_2d.mexw64 or KMAC_3d.mexw64,
% please contact Emmerich et al. and Yang et al. to obtain the C++ codes, and 
% then compile the codes into mex files via matlab.
%------------------------------- Reference --------------------------------
% [1] Emmerich, Michael, Kaifeng Yang, André Deutz, Hao Wang, and Carlos M. Fonseca. 
% "A multicriteria generalization of Bayesian global optimization." Advances in 
% stochastic and deterministic global optimization (2016): 229-242.
% [2] Yang, Kaifeng, Michael Emmerich, André Deutz, and Carlos M. Fonseca. 
% "Computing 3-D expected hypervolume improvement and related integrals in
% asymptotically optimal time." EMO2017, pp. 685-700.  

% This function was written by Liang Zhao.
% https://github.com/mobo-d/DirHV-EGO

    [test_mean,test_std]= GPEvaluate(test_x,GPModels); 
   
    if  n_obj == 2 % for 2 objs
        EHVI = KMAC_2d(-train_y_nds,-RefPoint,[-test_mean,test_std]);% with complexity of O(n log n)
    elseif  n_obj == 3 % for 3 objs
        EHVI = KMAC_3d(-train_y_nds,-RefPoint,[-test_mean,test_std]);% with complexity of O(n log n)
    else
        error("Infill_EHVI.m: The current code is only applicable for m=2 and m=3.");
    end
end
 
% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================
function [u,s] = GPEvaluate(X,model)
% Predict the GP posterior mean and std at a set of the candidate solutions 
    N = size(X,1); % number of samples
    M = length(model); % number of objectives
    u = zeros(N,M); % predictive mean
    MSE = zeros(N,M); % predictive MSE
    if N == 1 
        for j = 1 : M
            [u(:,j),~,MSE(:,j)] = Predictor(X,model{1,j}); % DACE Kriging toolbox
        end
        MSE(MSE<0) = 0;
    else
        for j = 1 : M
            [u(:,j),MSE(:,j)] = Predictor(X,model{1,j}); % DACE Kriging toolbox
        end
        MSE(MSE<0) = 0;
    end
   s = sqrt(MSE);% square root of the predictive variance
end
 