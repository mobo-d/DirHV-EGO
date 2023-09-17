function [SelectDecs] = Opt_DirHV_EI(Problem,GPModels,Objs_ND,q)
% Maximizing N Subproblems and Selecting Batch of Points 
% Expected Direction-based Hypervolume Improvement (DirHV-EI, denoted as EI_D)
 
%------------------------------- Reference --------------------------------
% L. Zhao and Q. Zhang, Hypervolume-Guided Decomposition for Parallel 
% Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
% Computation, 2023.
 
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
    if Problem.M <= 3
        [W, Problem.N]  = UniformPoint(Problem.N,Problem.M); % simplex-lattice design 
    else
        [W, Problem.N]  = UniformPoint(Problem.N,Problem.M,'ILD'); % incremental lattice design
    end
   %% Utopian point
     Z = -0.01.*ones([1,Problem.M]); % Adaptively adjusting Z may lead to better performance.
   %% Calculate the Intersection points and Direction vectors
     [Xi,Lambda] = CalXi(Objs_ND,W,Z);  
  
   %% Use MOEA/D to maximize DirHV-EI
    [~,PopDec,Pop_u,Pop_s] = MOEAD_GR_(Problem,Lambda,Xi,GPModels);
 
    %% Line 11 in Algorithm 2：discard the duplicate candidates
    [PopDec,ia,~] = unique(PopDec,'rows'); 
    Pop_u = Pop_u(ia,:); Pop_s = Pop_s(ia,:);

 
    %% Compute EI_D for all the points in Q
	L = size(PopDec,1); EIDs = zeros(L,Problem.N);
	for j = 1 : L
		EIDs(j,:) = EI_D_Cal(repmat(Pop_u(j,:),Problem.N,1),repmat(Pop_s(j,:),Problem.N,1),Xi); 
	end
	%% find q solutions with the greedy algorithm
	Batch_size = min(Problem.maxFE - Problem.FE,q); % the total budget is Problem.maxFE 
	Qb = Submodular_max(EIDs,Batch_size);  
 
    SelectDecs = PopDec(Qb,:); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Algorithm 2 & Algorithm 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function [Pop_EID,PopDec,Pop_u,Pop_s] = MOEAD_GR_(Problem,Lambda,Xi,GPModels)
%% Algorithm 2: using MOEA/D-GR to solve subproblems
    maxIter = 50;
    %% neighbourhood   
    T       = ceil(Problem.N/10); % size of neighbourhood: 0.1*N
    B       = pdist2(Lambda,Lambda);
    [~,B]   = sort(B,2);
    B       = B(:,1:T);
	
    % the initial population for MOEA/D
    PopDec = repmat(Problem.upper-Problem.lower,Problem.N,1).*UniformPoint(Problem.N,Problem.D,'Latin')+repmat(Problem.lower,Problem.N,1); 
    [Pop_u,Pop_s]= Evaluate(PopDec,GPModels); % calculate the predictive means and variances
    Pop_EID = EI_D_Cal(Pop_u,Pop_s,Xi); 
	
	% optimization
    for gen = 1 : maxIter-1
       for i = 1 : Problem.N    
           if rand < 0.8 % delta
               P = B(i,randperm(size(B,2)));
           else
               P = randperm(Problem.N);
           end
           %% generate an offspring and calculate its predictive mean and variance 
           OffDec = OperatorDE(Problem,PopDec(i,:),PopDec(P(1),:),PopDec(P(2),:)); 
           [Off_u,Off_s]= Evaluate(OffDec,GPModels);  
            
           %% Global Replacement  MOEA/D-GR
           % Find the most appropriate subproblem and its neighborhood
            EID_all = EI_D_Cal(repmat(Off_u,Problem.N,1),repmat(Off_s,Problem.N,1),Xi);
            [~,best_index] = max(EID_all);

            P = B(best_index,:); % replacement neighborhood
            % Update the solutions in P
           offindex = P(Pop_EID(P)<EID_all(P));
           if ~isempty(offindex)
               PopDec(offindex,:) = repmat(OffDec,length(offindex),1); 
               Pop_u(offindex,:) = repmat(Off_u,length(offindex),1);
               Pop_s(offindex,:) = repmat(Off_s,length(offindex),1);
               Pop_EID(offindex) = EID_all(offindex);
           end
       end   
    end
end

function Qb = Submodular_max(EIDs,q)
%% Algorithm 3: Submodularity-based Batch Selection
    [L,N] = size(EIDs);
    Qb=[]; temp = EIDs;
    beta = zeros([1,N]); 
    for i = 1 : q
        [~,index] = max(sum(temp,2));
        Qb = [Qb,index];
        beta = beta + temp(index,:);
        % temp: [EI_D(x|\lambda) - beta]_+
        temp = EIDs-repmat(beta,L,1);
        temp(temp < 0) = 0;   
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [Xi,Lambda] = CalXi(A,W,Z)  
% % calculate the minimum of mTch for each direction
% % L(A,w,z) = min max (A-z)./Lambda % used for Eq.11 
% % In:
% % A     : L*M  observed objectives 
% % W     : N*M  weight vectors
% % Z     : 1*M  reference point
% % Return:
% % Xi    : N*1  intersection points, one for each direction vector 
% % Lambda: N*M  direction vectors 

    [N,M] = size(W); % N: # of subproblems, M: # of objectives
    %% Eq. 25, the L2 norm of each direction vector should be 1
    W_ = 1.1.*W - repmat(Z,N,1);
    Lambda = W_./repmat(sqrt(sum(W_.^2,2)),1,M);
	
	%% Eq. 11, compute the intersection points
    Lambda_ = 1./Lambda;    A = A-Z; % L*M
    G = Lambda_(:,1)*A(:,1)'; % N*L, f1
    for j = 2:M
        G = max(G,Lambda_(:,j)*A(:,j)'); % N*L, max(fi,fj)
    end
    % minimum of mTch for each direction vector
    Lmin = min(G,[],2); % N*1  one for each direction vector 
    % N*M  Intersection points
    Xi = repmat(Z,N,1) + repmat(Lmin,1,M).*Lambda; 
end 


function EI_D = EI_D_Cal(u,sigma,Xis)
% calculate the EI_D(x|lambda) at multiple requests 
% u     : N*M  predictive mean
% sigma : N*M  square root of the predictive variance
% Xis   : N*M  Intersection points 
% % Eq. 23 in Proposition 5
    xi_minus_u = Xis-u; % N*M
    tau = xi_minus_u./sigma;  % N*M    
    temp = xi_minus_u.*normcdf(tau) + sigma.*normpdf(tau); % N*M 
    EI_D = prod(temp,2);
end
 
 
function [u,s] = Evaluate(X,model)
% Predict the objective vector of the candidate solutions 
    N = size(X,1); % number of samples
    M = length(model); % number of objectives
    u = zeros(N,M); % predictive mean
    MSE = zeros(N,M); % predictive SME
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
