classdef DirHVEGO < ALGORITHM
% <multi/many> <real> <expensive>
% DirHV-EGO: Multiobjective Efficient Global Optimization via Hypervolume-Guided Decomposition 
% batch_size    ---   5 --- number of true function evaluations per iteration
% n_init   ---   0 --- number of initial samples. A value of 0 indicates that n_init=11d-1. 

%------------------------------- Reference --------------------------------
% Liang Zhao and Qingfu Zhang, Hypervolume-Guided Decomposition for Parallel 
% Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
% Computation, 28(2): 432-444, 2024.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function was written by Liang Zhao.
% https://github.com/mobo-d/DirHV-EGO

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [batch_size, n_init] = Algorithm.ParameterSet(5,0); 
            % number of initial samples
            if n_init == 0, n_init = 11*Problem.D-1;end 
            
            %% Initial hyperparameters for GP
            GPModels = cell(1,Problem.M);   theta = cell(1,Problem.M);
            theta(:) = {(n_init ^ (-1./n_init)).*ones(1,Problem.D)};
            
            %% Generate initial design using LHS or other DOE methods
            %x_init = srgtsESEAOLHSdesign(C0,Problem.D,50,5);% OLHS (npoints, ndv,maxiter, maxstalliter);
            x_init = UniformPoint(n_init,Problem.D,'Latin'); %  LSH in PlatEMO
            Dn = Problem.Evaluation(repmat(Problem.upper-Problem.lower,n_init,1).*x_init+repmat(Problem.lower,n_init,1));     
            [FrontNo,~] = NDSort(Dn.objs,1); % find non-dominated solutions
			
            %% Optimization
            while Algorithm.NotTerminated(Dn(FrontNo==1))
              %% Line 3 in Algorithm 1：Scale the objective values 
                train_x = Dn.decs; D_objs = Dn.objs;
		        train_y = (D_objs-repmat(min(D_objs),size(train_x,1),1))./repmat(max(D_objs)-min(D_objs),size(train_x,1),1); 
                train_y_nds = train_y(FrontNo==1,:);
              %% Line 4 in Algorithm 1：Bulid GP model for each objective function 
                  % Note that if Problem.D > 8, it is recommended to use either the DE or Multi-start SQP methods 
                  % to optimize the hyperparameters of GP. Additionally, it is suggested to perform hyperparameter 
                  % optimization in the log scale (i.e., setting t=log \theta) to achieve better results.
                for i = 1 : Problem.M
                    GPModels{i}= Dacefit(train_x,train_y(:,i),'regpoly0','corrgauss',theta{i},1e-6*ones(1,Problem.D),20*ones(1,Problem.D));
                    theta{i}   = GPModels{i}.theta;
                end 
                
              %% Line 5 in Algorithm 1： Maximize DirHV-EI using the MOEA/D framework and select multiple candidate points
                new_x = Opt_DirHV_EI(Problem,GPModels,train_y_nds,batch_size); % batch_size*Problem.D
              
              %% Line 6 in Algorithm 1： Aggregate data
                Dn = [Dn,Problem.Evaluation(new_x)];
                [FrontNo,~] = NDSort(Dn.objs,1);
 
            end
        end
    end
end
