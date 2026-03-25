clear all;
clc;
folders = genpath('./Algorithms'); addpath(folders);
folders = genpath('./Metrics'); addpath(folders);
folders = genpath('./Problems'); addpath(folders);
  
n_run = 21; % # of runs
ins_list = {{'ZDT1',2,8,200},{'ZDT2',2,8,200},{'ZDT3',2,8,200},...
    {'ZDT4',2,8,200},{'ZDT6',2,8,200}, {'DTLZ1',3,6,300},...
    {'DTLZ2',3,6,300},{'DTLZ3',3,6,300},{'DTLZ4',3,6,300},...
    {'DTLZ5',3,6,300},{'DTLZ6',3,6,300},{'DTLZ7',3,6,300}};
 
for id = 1:1:length(ins_list)  
    clf;
    Problem = ins_list{id}; 
    prob_name = Problem{1,1}; 
    M = Problem{1,2}; D = Problem{1,3}; maxFE = Problem{1,4};
    score      = [];
    IGDps       = [];
    %% run DirHV-EGO
    for i = 1 : n_run
        Pro = feval(prob_name,'M',M,'D',D,'maxFE',maxFE); 
        Alg = DirHVEGO('save',Inf,'run',i,'metName',{'IGDp'});
        Alg.Solve(Pro);
        %% get metric value
        score = [score,Alg.metric.IGDp]; % list: IGDp,           
        IGDps = [IGDps;Alg.metric.IGDp(end)];
    end
    disp([sprintf('DirHV-EGO on %s_M%d_D%d_maxFE%d IGDp over %d runs:%.4e(%.2e)',prob_name,M,D,maxFE,n_run,mean(IGDps),std(IGDps))])
    % plot the convergence profile of logIGD+
    h = figure(i);
    FE   =  cell2mat(Alg.result(:,1))'; % list: # of evaluations 
    temp = log10(score);
    mean_logIGDp = mean(temp,2)';
    std_logIGDp = std(temp,0,2)';
    plot(FE,mean_logIGDp,'^b-','MarkerSize',7,'LineWidth',2);
    hold on;
    inBetween = [mean_logIGDp-std_logIGDp, fliplr(mean_logIGDp+std_logIGDp)];
    fill([FE, fliplr(FE)], inBetween, 'b','FaceAlpha',.3,'EdgeColor','none');
    set(gca,'XTick',fix(linspace(FE(1),FE(end),7)));  
    set(gca,'XLim',[FE(1)  FE(end)])
    set(gcf, 'Color', [1,1,1]);
    xlabel('Number of function evaluation');
    ylabel(append('log IGD+'));    
     title(sprintf('%s (DirHV-EGO)', prob_name));
    currentFile = mfilename('fullpath');
    filename = fullfile(fileparts(currentFile),sprintf('\\Data\\%s_%s_M%d_D%d_logIGDp',class(Alg),class(Pro),Pro.M,Pro.D));
    saveas(gcf,[filename,'.pdf'])
end
 
 
