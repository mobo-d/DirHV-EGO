classdef DirHVEGO < ALGORITHM
% <multi/many> <real> <expensive>
% DirHV-EGO: Multiobjective Efficient Global Optimization via Hypervolume-Guided Decomposition 
% q    ---   5 --- Batch size, i.e. number of function evaluations per iteration
% NI   ---   0 --- number of initial samples, If the parameter is 0, NI=11d-1. 

%------------------------------- Reference --------------------------------
% L. Zhao and Q. Zhang, Hypervolume-Guided Decomposition for Parallel 
% Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
% Computation, 2024, 28(2): 432-444.

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
            [q, NI] = Algorithm.ParameterSet(5,0); 
            if NI == 0, NI = 11*Problem.D-1;end % number of initial samples
            
            %% Initial hyperparameters for GP
            GPModels = cell(1,Problem.M);   theta = cell(1,Problem.M);
            theta(:) = {(NI ^ (-1./NI)).*ones(1,Problem.D)};
            
            %% Generate initial design using LHS or other DOE methods
            %X_init = srgtsESEAOLHSdesign(C0,Problem.D,50,5);% OLHS (npoints, ndv,maxiter, maxstalliter);
            X_init = UniformPoint(NI,Problem.D,'Latin'); %  LSH in PlatEMO
            Dn = Problem.Evaluation(repmat(Problem.upper-Problem.lower,NI,1).*X_init+repmat(Problem.lower,NI,1));     
            [FrontNo,~] = NDSort(Dn.objs,1); % find non-dominated solutions
			
            %% Optimization
            while Algorithm.NotTerminated(Dn(FrontNo==1))
              %% Line 3 in Algorithm 1：Scale the objective values 
                D_decs = Dn.decs; D_objs = Dn.objs;
		        D_objs_Scaled = (D_objs-repmat(min(D_objs),size(D_decs,1),1))./repmat(max(D_objs)-min(D_objs),size(D_decs,1),1); 
         
              %% Line 4 in Algorithm 1：Bulid GP model for each objective function 
                  % Note that if Problem.D > 8, it is recommended to use either the DE or Multi-start SQP methods 
                  % to optimize the hyperparameters of GP. Additionally, it is suggested to perform hyperparameter 
                  % optimization in the log scale (i.e., setting t=log \theta) to achieve better results.
                for i = 1 : Problem.M
                    GPModels{i}= Dacefit(D_decs,D_objs_Scaled(:,i),'regpoly0','corrgauss',theta{i},1e-6*ones(1,Problem.D),20*ones(1,Problem.D));
                    theta{i}   = GPModels{i}.theta;
                end 
                
              %% Line 5 in Algorithm 1： Maximize DirHV-EI using the MOEA/D framework and select q candidate points
                NewDecs = Opt_DirHV_EI(Problem,GPModels,D_objs_Scaled(FrontNo==1,:),q); % q*Problem.D
              
              %% Line 6 in Algorithm 1： Aggregate data
                Dn = [Dn,Problem.Evaluation(NewDecs)];
                [FrontNo,~] = NDSort(Dn.objs,1);
 
            end
        end
    end
end
