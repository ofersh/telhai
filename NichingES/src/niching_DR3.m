%Dynamic ES Niching with (1,lambda)-DR3.

function [X,mpr_q] = niching_DR3(bnf,N,X_a,X_b,q,q_eff,rho,kappa,co_sigma,NEC);
close all;
strfitnessfct = 'benchmark_func';
global initial_flag;
initial_flag=0;

% Strategy parameter setting: Selection
lambda=10; mu=1; 

%Strategy parameter setting: Adaptation
global alpha beta m c_m c c_u; 
alpha = 1.5;
beta = 1/sqrt(N);
m = round(1.5*N^2);
c_m = (m^-0.5)*(1+(1/m));
c = 1/sqrt(N);
c_u = sqrt((2-c)/c);

%Data-structures
X = (X_b-X_a)*rand(N,q_eff) + X_a; %decision parameters to be optimized.
Y = zeros(N,q_eff*lambda); %temporary DB for offspring.
P = zeros(1,q_eff*lambda); %Parents indices
% Initialize dynamic (internal) strategy parameters and constants
for i=1:q_eff,
    [delta{i},B{i}] = init_dr3(N,co_sigma);
end

gen = 0;
global_eval = 0;
arfitness = zeros(1,q_eff*lambda);
best = inf*ones(N+1,q);

out = 10;
MAX_EVAL = q*NEC;
MAX_GENERATIONS = ceil(MAX_EVAL/(q_eff*lambda));
stat = zeros(1,MAX_GENERATIONS);
mpr_q = zeros(q,MAX_GENERATIONS);
%step = zeros(q,MAX_GENERATIONS);

% -------------------- Generation Loop --------------------------------
while global_eval < MAX_EVAL %while gen < MAX_GENERATIONS
    arz = randn(m,q_eff*lambda);  % array of +normally distributed r.v.
    xi = alpha*(rand(1,q_eff*lambda)>=0.5); xi(find(xi==0))=(1/alpha);
    for k=1:q_eff*lambda,
        parent = ceil(k/lambda);
        Y(:,k) = X(:,parent) + xi(1,k)*delta{parent}*(c_m*B{parent}*arz(:,k));
        P(1,k) = parent;
    end

%%     Periodic Boundary Conditions - Let us keep X in the interval [X_a,X_b]:
%      if ((sum(sum(Y(:,:) < X_a)) > 0) || (sum(sum(Y(:,:) > X_b)) > 0))
%          Y(:,:) = Y(:,:).*(Y(:,:) >= X_a).*(Y(:,:) <= X_b) + X_a*(Y(:,:) < X_a) + X_b*(Y(:,:) > X_b);
%      end

    % Fitness evaluation + sorting
    arfitness(:) = (feval (strfitnessfct,Y(:,:)',bnf))';
    global_eval = global_eval + size(Y,2);
    [arfitness, arindex] = sort(arfitness,2,'ascend'); %  M I N I M I Z A T I O N
    Y = Y(:,arindex); % Decision+Strategy parameters are now sorted!
    arz = arz(:,arindex); xi = xi(:,arindex);
    P = P(:,arindex);

    stat(1,gen+1) = arfitness(1,1);
    MX=zeros(1,q);
    
    %Dynamic Peak Identification
    [DPS,pop_niche] = DPI (Y(:,:),lambda*q_eff,q,rho);

    %(1,lambda) Selection for each niche
    for i=1:q,
        %step(i,gen+1) = delta{i};
        j=DPS(1,i);
        if (j~=0)
            parent = P(1,j); %the original parent!
            X(:,i) = Y(:,j);
            [new_delta{i},new_B{i}] = ...
                dr3_adapt(delta{parent},B{parent},arz(:,j),xi(j),N);
        else
            X(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            [new_delta{i},new_B{i}] = init_dr3(N,co_sigma);
        end
    end
    if (mod(gen,kappa)==0)
        for i=q+1:q_eff,
            X(:,i) = (X_b-X_a)*rand(N,1) + X_a;
            [new_delta{i},new_B{i}] = init_dr3(N,co_sigma);
        end
    end
    delta = new_delta;
    B = new_B;
    MX = (feval(strfitnessfct,X(:,1:q)',bnf))';

    % Output
%     if (mod(gen,out)==0)
%         disp([num2str(gen) ': ' num2str(MX(:,:))]);
%     end

    mpr_q(:,gen+1) = MX;%1./(1+abs(MX(:,:)'));
    gen = gen + 1;
end

X = X(:,1:q);
[best(N+1,:), arindex] = sort(best(N+1,:),2,'ascend'); % minimization
best(1:N,:) = best(1:N,arindex);

disp([num2str(gen) ': ' num2str(MX(:,:))]);

%--------------------------------------------------------------------------
function[delta,B] = init_dr3(N,co_sigma);
global c m;
delta = co_sigma;
B = [zeros(N,1),c*randn(N,m-1)];
%--------------------------------------------------------------------------
function[new_delta,new_B] = dr3_adapt(delta,B,arz,xi,N);
global beta c c_u c_m m;
new_delta = delta*(xi^beta);
b = (1-c)*B(:,1) + c*(c_u*xi*(c_m*B*arz));
new_B = [b,B(:,1:(m-1))];
    
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
%--------------------------------------------------------------------------