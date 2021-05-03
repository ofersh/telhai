%Dynamic ES Niching with (1+lambda)-CMA.
% -----------------------------------------------------------------------------------
% Niching CMA-ES: Dynamic Niching with Covariance Matrix Adaptation ES
% for nonlinear function multimodal minimization. To be used under the
% terms of the GNU General Public License:
% http://www.gnu.org/copyleft/gpl.html
% CMA+ implementation is based on Igel et al., GECCO-2006 ACM paper.
%
% Author: Ofer M. Shir, 2006. e-mail: oshir@liacs.nl
% http://www.liacs.nl/~oshir/
% -----------------------------------------------------------------------------------

function [X,mpr_q] = niching_cmaplus(strfitnessfct,N,X_a,X_b,...
    q,q_eff,rho,kappa,co_sigma,MAX_EVAL);
close all;

%Strategy parameter setting: Selection
mu = 1; lambda=10;

%Strategy parameter setting: Adaptation
ptarget = (2/11); pts = (ptarget/(1-ptarget));
pthresh = 0.44;
cc = 2/(N+2);
ccov = 2/(6+N^2);
d = 1+(N/2);
cp = (1/12);

%Data-structures
X = (X_b-X_a)*rand(N,q_eff) + X_a; %decision parameters to be optimized.
Y = zeros(N,q_eff*lambda); %temporary DB for offspring.
P = zeros(1,q_eff*lambda); %Parents indices
% Initialize dynamic (internal) strategy parameters and constants
for i=1:q_eff,
    [sigma{i},pc{i},ps{i},B{i},D{i},C{i},counteval{i}] = ...
        init_cma(N,co_sigma,ptarget);
end

gen = 0;
global_eval = 0;
arfitness = inf*ones(1,q_eff*(lambda+1)); %Offspring Fitness
arP = feval (strfitnessfct,X(:,:)); %Parents' Fitness

out = 10;
MAX_GENERATIONS = ceil(MAX_EVAL/(q_eff*lambda));
stat = zeros(1,MAX_GENERATIONS);
mpr_q = zeros(q,MAX_GENERATIONS);

% -------------------- Generation Loop --------------------------------
while global_eval < MAX_EVAL
    Y = zeros(N,q_eff*(lambda+1)); %temporary DB for offspring.
    P = zeros(1,q_eff*(lambda+1)); %Parents indices
    arz = randn(N,q_eff*lambda);  % array of normally distributed r.v.
    for k=1:q_eff*lambda,
        parent = ceil(k/lambda);
        Y(:,k) = X(:,parent) + sigma{parent}*(B{parent}*...
            D{parent}*arz(:,k)); % add mutation
        if ((sum(Y(:,k) < X_a) > 0) || (sum(Y(:,k) > X_b) > 0))
            Y(:,k) = Y(:,k).*(Y(:,k) >= X_a).*(Y(:,k) <= X_b) +...
                X_a*(Y(:,k) < X_a) + X_b*(Y(:,k) > X_b);
        end
        arfitness(k) = feval (strfitnessfct,Y(:,k));
        global_eval = global_eval + 1;
        counteval{parent} = counteval{parent}+1;
        P(1,k) = parent;
    end

    %Concatenation of the parents to the DB:
    arz = [arz,zeros(N,q_eff)];
    arfitness(:,q_eff*lambda+1:end) = arP;
    P(:,q_eff*lambda+1:end) = [1:1:q_eff];
    Y(:,q_eff*lambda+1:end) = X(:,:);

    % Fitness sorting
    [arfitness, arindex] = sort(arfitness,2,'ascend'); %  M I N I M I Z A T I O N
    Y = Y(:,arindex); % Decision+Strategy parameters are now sorted!
    arz = arz(:,arindex);
    P = P(:,arindex);

    stat(1,gen+1) = arfitness(1,1);

    %Dynamic Peak Identification
    [DPS,pop_niche] = DPI (Y(:,:),(lambda+1)*q_eff,q,rho);

    %(1+lambda) Selection for each niche
    FitP = inf*ones(1,q_eff);
    XP = zeros(N,q_eff);
    for i=1:q,
        j=DPS(1,i);
        if (j~=0)
            parent = P(1,j); %the original parent!
            XP(:,i) = Y(:,j);
            FitP(1,i) = arfitness(1,j);
            lsucc = (arfitness(1,j)<arP(1,parent)); %Recent Mutation Success ( m i n i m i z a t i o n )
            [new_sigma{i},new_ps{i},new_counteval{i}] = ...
                updateStepSize(sigma,ps,counteval,lsucc,cp,d,pts,parent);
            if (lsucc)
                [new_pc{i},new_B{i},new_D{i},new_C{i}] = ...
                    updateCov(arz(:,j),C,B,D,pc,new_ps{i},parent,cc,ccov,pthresh);
            else
                new_pc{i} = pc{parent};
                new_B{i} = B{parent};
                new_D{i} = D{parent};
                new_C{i} = C{parent};
            end
        else
            XP(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},new_C{i},...
                new_counteval{i}] = init_cma(N,co_sigma,ptarget);
            FitP(1,i) = feval (strfitnessfct,XP(:,i));
        end
    end
    if (mod(gen,kappa)==0)
        for i=q+1:q_eff,
            XP(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            [new_sigma{i},new_pc{i},new_ps{i},new_B{i},new_D{i},...
                new_C{i},new_counteval{i}] = init_cma(N,co_sigma,ptarget);
            FitP(1,i) = feval (strfitnessfct,XP(:,i));
        end
    end
    sigma = new_sigma;
    pc = new_pc;
    ps = new_ps;
    B = new_B;
    C = new_C;
    D = new_D;
    counteval = new_counteval;
    X = XP;
    arP = FitP;

    mpr_q(:,gen+1) = arP(:,1:q);
    % Output
%         if (mod(gen,out)==0)
%             disp([num2str(gen) ': ' num2str(arP(:,1:q))]);
%         end

    gen = gen + 1;
end

X = X(:,1:q);

disp([num2str(gen) ': ' num2str(arP(:,:))]);
end
%--------------------------------------------------------------------------
function[new_sigma,new_ps,new_counteval] = updateStepSize(sigma,ps,counteval,lsucc,cp,d,pts,parent);
new_ps = (1-cp)*ps{parent} + cp*lsucc;
new_sigma = sigma{parent}*exp((1/d)*(new_ps-(pts*(1-new_ps))));
new_counteval = counteval{parent};
end
%--------------------------------------------------------------------------
function[new_pc,new_B,new_D,new_C] = updateCov(Z,C,B,D,pc,ps,parent,cc,ccov,pthresh);
if (ps < pthresh)
    new_pc = (1-cc)*pc{parent} + sqrt(cc*(2-cc))*(B{parent}*D{parent}*Z);
    new_C = (1-ccov)*C{parent} + ccov*(new_pc*new_pc');
else
    new_pc = (1-cc)*pc{parent};
    new_C = (1-ccov)*C{parent} + ccov*(new_pc*new_pc' + cc*(2-cc)*C{parent});
end
% Update B and D from C
N=size(C,1);
new_C=triu(new_C)+triu(new_C,1)'; % enforce symmetry
[new_B,new_D] = eig(new_C);       % eigen decomposition, B==normalized eigenvectors
new_D(new_D<1E-10)=1E-10; 
if max(diag(new_D)) > 1E14*min(diag(new_D))
    tmp = max(diag(new_D))/1E14 - min(diag(new_D));
	new_C = new_C + tmp*eye(N); new_D = new_D + tmp*eye(N); 
end
new_D = abs(diag(sqrt(diag(new_D)))); % D contains standard deviations now
end
%--------------------------------------------------------------------------
function[sigma,pc,ps,B,D,C,counteval] = init_cma(N,co_sigma,ptarget);
pc = zeros(N,1); ps = ptarget;      % evolution paths for C and sigma
B = eye(N);                         % B defines the coordinate system
D = eye(N);                         % diagonal matrix D defines the scaling
C = B*D*(B*D)';                     % covariance matrix
sigma = co_sigma;
counteval = 0;
end
%--------------------------------------------------------------------------
function [DPS,pop_niche] = DPI (Y,psize,q,rho);
DPS = zeros(1,q); %Dynamic Peak Set.
pop_niche = zeros(1,psize); %The classification of each individual to a niche; zero is "non-peak" domain.
Num_Peaks = 1;
niche_count = zeros(1,q);
DPS(1,1) = 1;
niche_count(1,1) = 1;
pop_niche(1,1) = 1;

for k=2:psize,
    assign = 0;
    for j=1:Num_Peaks,
        d_pi = Y(:,k) - Y(:,DPS(1,j));
        %if (norm(d_pi) < y_rho(1,DPS(1,j)))
        if (norm(d_pi) < rho)
            niche_count(1,j) = niche_count(1,j)+1;
            pop_niche(1,k) = j;
            assign = 1;
            break;
        end
    end
    if ((Num_Peaks<q)&&(assign==0))
        Num_Peaks = Num_Peaks + 1;
        DPS(1,Num_Peaks) = k;
        niche_count(1,Num_Peaks) = 1;
        pop_niche(1,k) = Num_Peaks;
    end
end
pop_niche(find(pop_niche==0)) = q+1;
end
%--------------------------------------------------------------------------
