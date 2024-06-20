classdef EHVIEGO < ALGORITHM
% <multi> <real/integer> <expensive>
% EHVI-EGO

%------------------------------- Reference --------------------------------
% Emmerich, Michael TM, Kyriakos C. Giannakoglou, and Boris Naujoks. 
% "Single-and multiobjective evolutionary optimization assisted by 
% Gaussian random field metamodels." IEEE Transactions on Evolutionary 
% Computation 10, no. 4 (2006): 421-439. 
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function was written by Liang Zhao.
% https://github.com/mobo-d/R2D-EGO

    methods
        function main(Algorithm,Problem)
           %% Parameter setting
            % number of initial samples
            n_init = 11*Problem.D-1;
            % Initial hyperparameters for GP
            theta = repmat({(n_init ^ (-1 ./ n_init)) .* ones(1, Problem.D)}, 1, Problem.M);
            
            %% Generate initial design using LHS or other DOE methods
            x_lhs = lhsdesign(n_init, Problem.D,'criterion','maximin','iterations',1000);
            x_init = Problem.lower +  (Problem.upper - Problem.lower).*x_lhs;  
            Archive = Problem.Evaluation(x_init);     
            % find non-dominated solutions
            FrontNo = NDSort(Archive.objs,1); 
 
            %% Optimization
            while Algorithm.NotTerminated(Archive(FrontNo==1))
                % scale the objective values to [0, 1]
                train_x = Archive.decs; ori_objs = Archive.objs;
                ymin    = min(ori_objs,[],1); ymax = max(ori_objs,[],1);
		        train_y = (ori_objs-ymin)./(ymax - ymin);
                train_y_nds = train_y(FrontNo==1,:);
                
              %% Bulid GP model for each objective function 
                 GPModels = cell(1,Problem.M);
                 for i = 1 : Problem.M
                    GPModels{i}= Dacefit(train_x,train_y(:,i),'regpoly0','corrgauss',theta{i},1e-6*ones(1,Problem.D),20*ones(1,Problem.D));
                    theta{i}   = GPModels{i}.theta;
                end 
              %% maximize EHVI using EA
                pop_size = Problem.N; maxIter = 50; 
                RefPoint = 1.1*ones(1, Problem.M);
                infill_criterion = @(x)-Infill_EHVI(x,GPModels,train_y_nds,RefPoint,Problem.M);% minimize -EHVI(x)
                new_x = Optimizer_DE(infill_criterion, Problem.D, Problem.lower,Problem.upper,pop_size,maxIter);
                %%  Expensive Evaluation
                Archive = [Archive,Problem.Evaluation(new_x)];
                FrontNo = NDSort(Archive.objs,1);
            end
        end
    end
end