classdef DirHVEGO < ALGORITHM
% <multi/many> <real> <expensive>
% DirHV-EGO: Multiobjective Efficient Global Optimization via Hypervolume-Guided Decomposition 
% q    ---   5 --- Batch size  

%------------------------------- Reference --------------------------------
% L. Zhao and Q. Zhang, Hypervolume-Guided Decomposition for Parallel 
% Expensive Multiobjective Optimization. IEEE Transactions on Evolutionary 
% Computation, 2023.

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
            q = Algorithm.ParameterSet(5);
            C0 = 11*Problem.D-1; % number of initial samples
            
            %% Initial hyperparameters for GP
            GPModels = cell(1,Problem.M);   theta = cell(1,Problem.M);
            theta(:) = {(C0 ^ (-1./C0)).*ones(1,Problem.D)};
            
            %% Generate initial design using OLHS or other DOE methods
            Decs = srgtsESEAOLHSdesign(C0,Problem.D,50,5);% OLHS (npoints, ndv,maxiter, maxstalliter);
            %Decs = UniformPoint(C0,Problem.D,'Latin'); %  LSH in PlatEMO
            D_pop = Problem.Evaluation(repmat(Problem.upper-Problem.lower,C0,1).*Decs+repmat(Problem.lower,C0,1));     
            [FrontNo,~] = NDSort(D_pop.objs,1); % find non-dominated solutions
			
            %% Optimization
            while Algorithm.NotTerminated(D_pop(FrontNo==1))
              %% Line 3 in Algorithm 1：Scale the objective values 
                D_decs = D_pop.decs; D_objs = D_pop.objs;
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
                 SelectDecs = Opt_DirHV_EI(Problem,GPModels,D_objs_Scaled(FrontNo==1,:),q); 
              
              %% Line 6 in Algorithm 1： Aggregate data
                D_pop = [D_pop,Problem.Evaluation(SelectDecs)];
                [FrontNo,~] = NDSort(D_pop.objs,1);
 
            end
        end
    end
end
