classdef KBEHVIEGO < ALGORITHM
% <multi> <real/integer> <expensive>
% Kriging Believer&EHVI-EGO (KB&EHVI-EGO)
% batch_size --- 5 --- number of true function evaluations per iteration 

%------------------------------- Reference --------------------------------
% Wada, Takashi, and Hideitsu Hino. "Bayesian optimization for multi-
% objective optimization and multi-point search." arXiv preprint arXiv:
% 1905.02370 (2019). https://arxiv.org/abs/1905.02370

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
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
            batch_size = Algorithm.ParameterSet(5); 
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
                 train_x = Archive.decs; D_objs = Archive.objs;
                 NewDecs = []; Batch_size = min(Problem.maxFE - Problem.FE,batch_size); 
                 for j =  1:1:Batch_size    
                     % scale the objective values to 0 and 1
                    ymin    = min(D_objs,[],1); ymax = max(D_objs,[],1);
		            train_y = (D_objs-ymin)./(ymax - ymin);
                    FrontN = NDSort(train_y,1);
                    train_y_nds = train_y(FrontN==1,:);
                    % bulid GP models for all the objective functions
                    GPModels = cell(1,Problem.M);
                    for i = 1 : Problem.M
                        GPModels{i}= Dacefit(train_x,train_y(:,i),'regpoly0','corrgauss',theta{i},1e-6*ones(1,Problem.D),20*ones(1,Problem.D));
                        theta{i} = GPModels{i}.theta;
                    end 
                    % select one candidate with the maximum EHVI value using DE
                    pop_size = Problem.N; maxIter = 50; RefPoint = 1.1*ones(1,Problem.M);
                    infill_criterion = @(x)-Infill_EHVI(x,GPModels,train_y_nds,RefPoint,Problem.M);% minimize -EHVI(x)
                    infill_x = Optimizer_DE(infill_criterion, Problem.D, Problem.lower,Problem.upper,pop_size,maxIter);
                    infill_y = zeros(1, Problem.M);
                    for i = 1 : Problem.M
                        infill_y(1,i) = Predictor(infill_x,GPModels{i}); % DACE Kriging toolbox
                    end
                    infill_y = ymin + infill_y.*(ymax-ymin);
                    NewDecs     = [NewDecs;infill_x];
                    D_objs = [D_objs;infill_y]; train_x = [train_x;infill_x];
                  end
                %% Expensive Evaluation
                Archive = [Archive,Problem.Evaluation(NewDecs)];
                FrontNo = NDSort(Archive.objs,1);
            end
        end
    end
end