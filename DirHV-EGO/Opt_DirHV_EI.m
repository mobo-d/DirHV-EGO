function [new_x] = Opt_DirHV_EI(Problem,GPModels,train_y_nds,batch_size)
% Maximizing N Subproblems and Selecting Batch of Query Points 
% Expected Direction-based Hypervolume Improvement (DirHV-EI, denoted as EI_D)
 
%------------------------------- Reference --------------------------------
% Liang Zhao and Qingfu Zhang, Hypervolume-Guided Decomposition for Parallel 
% Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
% Computation, 28(2): 432-444, 2024.
 
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function was written by Liang Zhao.
% https://github.com/mobo-d/DirHV-EGO

 
   %% Generate the initial direction vectors
   params_N =  [200,210,295,456,462]; % # of weight vectors for Problem.M = 2,3,4,5,6 respectively.
    if Problem.M <= 3
        [ref_vecs, Problem.N]  = UniformPoint(params_N(Problem.M-1),Problem.M); % simplex-lattice design 
    elseif Problem.M <= 6
        [ref_vecs, Problem.N]  = UniformPoint(params_N(Problem.M-1),Problem.M,'ILD'); % incremental lattice design
    else
        [ref_vecs, Problem.N]  = UniformPoint(400,Problem.M,'ILD'); % incremental lattice design
        %disp('Warning: The computational complexity of DirHV-EGO is quadratic to the number of reference vectors!');
    end
   %% Utopian point
     z = -0.01.*ones([1,Problem.M]); % Adaptively adjusting Z may lead to better performance.

   %% Calculate the Intersection points and Direction vectors
     [xis,dir_vecs] = get_xis(train_y_nds,ref_vecs,z);  
  
   %% Use MOEA/D to maximize DirHV-EI
    [~,candidate_x,candidate_mean,candidata_std] = MOEAD_GR_(Problem,dir_vecs,xis,GPModels);
 
    %% Line 11 in Algorithm 2：discard the duplicate candidates
    [candidate_x,ia,~] = unique(candidate_x,'rows'); 
    candidate_mean = candidate_mean(ia,:); candidata_std = candidata_std(ia,:);
 
    %% Compute EI_D for all the points in Q
    EIDs = zeros(size(candidate_x,1),size(dir_vecs,1));
    for j = 1 : size(candidate_x,1)
	    EIDs(j,:) = get_EID(repmat(candidate_mean(j,:),size(dir_vecs,1),1),repmat(candidata_std(j,:),size(dir_vecs,1),1),xis); 
    end
    %% find q solutions with the greedy algorithm
    % the total budget is Problem.maxFE 
    Qb = subset_selection(EIDs,min(Problem.maxFE - Problem.FE,batch_size));  
    new_x = candidate_x(Qb,:); 
end

% >>>>>>>>>>>>>>>>   Algorithm 2 & Algorithm 3  ====================
function [pop_EID,pop_x,pop_mean,pop_std] = MOEAD_GR_(Problem,dir_vecs,xis,GPModels)
%% Algorithm 2: using MOEA/D-GR to solve subproblems
    % In order to find the maximum value of DirHV-EI for each sub-problem, 
    % it is recommended to set the maximum number of iterations to at least 50.
    maxIter = 50; 
    pop_size = size(dir_vecs,1);
    %% neighbourhood   
    T       = ceil(pop_size/10); % size of neighbourhood: 0.1*N
    B       = pdist2(dir_vecs,dir_vecs);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);
	
    % the initial population for MOEA/D
    pop_x = repmat(Problem.upper-Problem.lower,pop_size,1).*UniformPoint(pop_size,Problem.D,'Latin')+repmat(Problem.lower,pop_size,1); 
    [pop_mean,pop_std]= GPEvaluate(pop_x,GPModels); % calculate the predictive means and variances
    pop_EID = get_EID(pop_mean,pop_std,xis); 
	
	% optimization
    for gen = 1 : maxIter-1
       for i = 1 : pop_size    
           if rand < 0.8 % delta
               P = B(i,randperm(size(B,2)));
           else
               P = randperm(pop_size);
           end
           %% generate an offspring and calculate its predictive mean and variance 
           off_x = OperatorDE(Problem,pop_x(i,:),pop_x(P(1),:),pop_x(P(2),:)); 
           [off_mean,off_std]= GPEvaluate(off_x,GPModels);  
            
           %% Global Replacement  MOEA/D-GR
           % Find the most appropriate subproblem and its neighborhood
            EID_all = get_EID(repmat(off_mean,pop_size,1),repmat(off_std,pop_size,1),xis);
            [~,best_index] = max(EID_all);

            P = B(best_index,:); % replacement neighborhood
            % Update the solutions in P
            offindex = P(pop_EID(P)<EID_all(P));
           if ~isempty(offindex)
               pop_x(offindex,:) = repmat(off_x,length(offindex),1); % PopDec: N*D
               pop_mean(offindex,:) = repmat(off_mean,length(offindex),1); % Pop_u: N*M
               pop_std(offindex,:) = repmat(off_std,length(offindex),1); % Pop_s: N*M
               pop_EID(offindex) = EID_all(offindex); % Pop_EID: N*1
           end
       end   
    end
end

function Qb = subset_selection(EIDs,Batch_size)
%% Algorithm 3: Submodularity-based Batch Selection
    [L,N] = size(EIDs);
    Qb=[]; temp = EIDs;
    beta = zeros([1,N]); 
    for i = 1 : Batch_size
        [~,index] = max(sum(temp,2));
        Qb = [Qb,index];
        beta = beta + temp(index,:);
        % temp: [EI_D(x|\lambda) - beta]_+
        temp = EIDs-repmat(beta,L,1);
        temp(temp < 0) = 0;   
    end
end

% >>>>>>>>>>>>>>>>   Auxiliary functions  ====================
function [xis,dir_vecs] = get_xis(train_y_nds,ref_vecs,z)  
% % calculate the minimum of mTch for each direction
% % L(A,w,z) = min max (A-z)./Lambda % used for Eq.11 
% % In:
% % train_y_nds  : n*M  observed objectives 
% % ref_vecs     : N*M  weight vectors
% % z            : 1*M  reference point
% % Return:
% % xis          : N*1  intersection points, one for each direction vector 
% % dir_vecs     : N*M  direction vectors 

    [N,M] = size(ref_vecs); % N: # of subproblems, M: # of objectives
    %% Eq. 25, the L2 norm of each direction vector should be 1
    temp = 1.1.*ref_vecs - repmat(z,N,1);
    dir_vecs = temp./repmat(sqrt(sum(temp.^2,2)),1,M);
	
	%% Eq. 11, compute the intersection points
    div_dir = 1./dir_vecs; % N*M
    train_y_nds = train_y_nds-z; % n*M
    G = div_dir(:,1)*train_y_nds(:,1)'; % N*n, f1
    for j = 2:M
        G = max(G,div_dir(:,j)*train_y_nds(:,j)'); % N*n, max(fi,fj)
    end
    % minimum of mTch for each direction vector
    Lmin = min(G,[],2); % N*1  one for each direction vector 
    % N*M  Intersection points
    xis = repmat(z,N,1) + repmat(Lmin,1,M).*dir_vecs; % Eq.11
end 


function EI_D = get_EID(u,sigma,xis)
% calculate the EI_D(x|lambda) at multiple requests 
% u     : N*M  predictive mean
% sigma : N*M  square root of the predictive variance
% Xis   : N*M  Intersection points 
% % Eq. 23 in Proposition 5
    xi_minus_u = xis-u; % N*M
    tau = xi_minus_u./sigma;  % N*M    
    temp = xi_minus_u.*normcdf(tau) + sigma.*normpdf(tau); % N*M 
    EI_D = prod(temp,2);
end
 
 
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
