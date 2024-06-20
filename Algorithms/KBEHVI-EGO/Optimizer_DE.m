function [best_x, best_y] = Optimizer_DE(obj_fun, num_vari, lower_bound, upper_bound, pop_size, max_gen)
% this is the DE (differential evolution) algorithm  for minimization
% problem
% Note: calculate f(X) at multiple requests
% F              DE-stepsize F from interval [0, 2]
% CR             crossover probability constant from interval [0, 1]
% strategy       1 --> DE/best/1/exp           6 --> DE/best/1/bin
%                2 --> DE/rand/1/exp           7 --> DE/rand/1/bin
%                3 --> DE/rand-to-best/1/exp   8 --> DE/rand-to-best/1/bin
%                4 --> DE/best/2/exp           9 --> DE/best/2/bin
%                5 --> DE/rand/2/exp           else  DE/rand/2/bin
 
    %% Parameter setting
    F = 0.8; CR = 0.8; strategy = 6;
    refresh = 0; % print the information to screen
       
    % Initialize population and some arrays
    pop_decs = lhsdesign(pop_size, num_vari).*(upper_bound - lower_bound) + lower_bound;
    
    % Evaluate the best member after initialization
    pop_fitness = feval(obj_fun, pop_decs);
    [best_y, ibest] = min(pop_fitness);
    bestmemit = pop_decs(ibest,:);% best member of current iteration
    best_x   = bestmemit; % best member ever

    %------------------------------DE-Minimization-------------------------------------
    % popold is the population which has to compete. It is static through one iteration. 
    % pop is the newly emerging population.
    % initialize bestmember  matrix
    bm  = zeros(pop_size,num_vari);
    % intermediate population of perturbed vectors
    ui  = zeros(pop_size,num_vari);
    % rotating index array (size pop_size)
    rot = (0:1:pop_size-1);
    % rotating index array (size D)
    rotd= (0:1:num_vari-1);
    lower_bound = repmat(lower_bound, pop_size,1);
    upper_bound = repmat(upper_bound, pop_size,1);
    iter = 1;
    while iter < max_gen
        % save the old population
        popold = pop_decs;
        % index pointer array
        ind = randperm(4);
        % shuffle locations of vectors
        a1  = randperm(pop_size);
        % rotate indices by ind(1) positions
        rt = rem(rot+ind(1),pop_size);
        % rotate vector locations
        a2  = a1(rt+1);
        rt = rem(rot+ind(2),pop_size);
        a3  = a2(rt+1);
        rt = rem(rot+ind(3),pop_size);
        a4  = a3(rt+1);
        rt = rem(rot+ind(4),pop_size);
        a5  = a4(rt+1);
        % shuffled population 1
        pm1 = popold(a1,:);
        % shuffled population 2
        pm2 = popold(a2,:);
        % shuffled population 3
        pm3 = popold(a3,:);
        % shuffled population 4
        pm4 = popold(a4,:);
        % shuffled population 5
        pm5 = popold(a5,:);
        % population filled with the best member of the last iteration
        for i=1:pop_size
            bm(i,:) = bestmemit;
        end
        % all random numbers < CR are 1, 0 otherwise
        mui = rand(pop_size,num_vari) < CR;
        % binomial crossover
        if (strategy > 5)
            st = strategy-5;
        else
            % exponential crossover
            st = strategy;
            % transpose, collect 1's in each column
            mui=sort(mui,2)';
            for i=1:pop_size
                n=floor(rand*num_vari);
                if n > 0
                    rtd = rem(rotd+n,num_vari);
                    %rotate column i by n
                    mui(:,i) = mui(rtd+1,i);
                end
            end
            % transpose back
            mui = mui';
        end
        % inverse mask to mui
        mpo = mui < 0.5;
        
        % strategy       1 --> DE/best/1/exp                     6 --> DE/best/1/bin
        %                        2 --> DE/rand/1/exp                    7 --> DE/rand/1/bin
        %                        3 --> DE/rand-to-best/1/exp     8 --> DE/rand-to-best/1/bin
        %                        4 --> DE/best/2/exp                     9 --> DE/best/2/bin
        %                        5 --> DE/rand/2/exp                     else  DE/rand/2/bin
        
        switch st
            % DE/best/1
            case 1
                % differential variation
                ui = bm + F*(pm1 - pm2);
                % crossover
                ui = popold.*mpo + ui.*mui;
                % DE/rand/1
            case 2
                % differential variation
                ui = pm3 + F*(pm1 - pm2);
                % crossover
                ui = popold.*mpo + ui.*mui;
                % DE/rand-to-best/1
            case 3
                ui = popold + F*(bm-popold) + F*(pm1 - pm2);
                % crossover
                ui = popold.*mpo + ui.*mui;
                % DE/best/2
            case 4
                % differential variation
                ui = bm + F*(pm1 - pm2 + pm3 - pm4);
                % crossover
                ui = popold.*mpo + ui.*mui;
                % DE/rand/2
            case 5
                % differential variation
                ui = pm5 + F*(pm1 - pm2 + pm3 - pm4);
                % crossover
                ui = popold.*mpo + ui.*mui;
        end
        
        % correcting violations on the lower bounds of the variables
        % these are good to go
        maskLB = ui > lower_bound;
        % these are good to go
        maskUB = ui < upper_bound;
        ui     = ui.*maskLB.*maskUB + lower_bound.*(~maskLB) + upper_bound.*(~maskUB);
        
        %%-------------------------------------------------------------------------
        % Select which vectors are allowed to enter the new population
        tempval = feval(obj_fun, ui);
        % if competitor is better than value in "cost array"
        indx = tempval <= pop_fitness;
        % replace old vector with new one (for new iteration)
        pop_decs(indx, :) = ui(indx, :);
        pop_fitness(indx, :) = tempval(indx, :);
        % we update bestval only in case of success to save time
        indx = tempval < best_y;
        if sum(indx)~=0
            [best_y, ind] = min(tempval);
            best_x = ui(ind,:);
        end
      
        % freeze the best member of this iteration for the coming
         % iteration. This is needed for some of the strategies.
        bestmemit = best_x;
        % print the information to screen
        if refresh == 1
            fprintf('Iteration: %d,  Best: %f,  F: %f,  CR: %f,  NP: %d\n',iter,best_y,F,CR,pop_size);
        end    
        iter = iter + 1;
    end %---end while ((iter < itermax) ...
end
