% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%% 
% %
% % Linear PAIR Example: Heat Example
% %
% %%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
% Creates linear PAIR for a toy heat problem.  Calculates and compares 
% singular valules for different latent/full maps.  Visualizes PAIR 
% results on 1 dimensional example.
% %%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

%% Create Data
rng(21)

N = 50;                  % Size 50 x 1 Input/Target
A = heat(N);             % Heat Example
% [A,~,~] = svd(A);      % Orthogonal Example
% A = randn(N,N);        % Invertible Example
% A = A'*A;              % Invertible SPD Example
K = 1000;              % Number of Samples
 
t = linspace(0,1,N)'/2; Xtrue = zeros(N,K); Bnoisy = zeros(N,K); Btrue = zeros(N,K);
minNoise = 1e-4; maxNoise = 1e-4;   % Add Noise
% minNoise = 0; maxNoise = 0;         % No Noise
noiseLevel = minNoise + (maxNoise-minNoise)*rand(1,K);

for k = 1:K
      r = randi(10,3) - 0.5*rand(3,1);
      x = sin(r(1)*2*pi*t) + sin(r(2)*2*pi*t);
      Xtrue(:,k) = x + abs(min(x));
      Xtrue(:,k) = rand(N,1);     % Full Rank Data Matrix
  Btrue(:,k) = A*Xtrue(:,k);
%   Noise(:,k)
  Bnoisy(:,k) = Btrue(:,k) + noiseLevel(k)*randn(N,1);
end
Adagger = pinv(A);
sigma = 1e-4;
Bmu = mean(Bnoisy,2);
Bcov = cov(Bnoisy')+sigma*eye(N);
Xmu = mean(Xtrue,2);
Xcov = cov(Xtrue')+sigma*eye(N);
LB = chol(Bcov)';
LX = chol(Xcov)';
NB = [LB Bmu];
NX = [LX Xmu];


%% Take Necessary SVDs
[UX, SX, VX] = svd(Xtrue);
[UB, SB, VB] = svd(Bnoisy);
[UA, SA, VA] = svd(A);
[UAdag, SAdag, VAdag] = svd(Adagger);

%% For Different Ranks, Construct AEs and PAIR Mappings 
for r=1:N
%     Mf_{r} = UB(:,1:r)' * A * NX * pinv(UX(:,1:r)'*NX);
%     Mi_{r} = UX(:,1:r)' * (A\(NB*pinv(UB(:,1:r)'*NB)));
    Mf_{r} = SB(1:r,:)*VB'*VX*[diag(1./diag(SX(1:r,1:r)));zeros(K-r,r)];
    Mi_{r} = SX(1:r,:)*VX'*VB*[diag(1./diag(SB(1:r,1:r)));zeros(K-r,r)];
    [~, Mf_S{r}, ~] = svd(Mf_{r});
    [~, Mi_S{r}, ~] = svd(Mi_{r});
    FOR_{r} = UB(:,1:r) * Mf_{r} * UX(:,1:r)';
    INV_{r} = UX(:,1:r) * Mi_{r} * UB(:,1:r)';
    [~, FOR_S{r}, ~] = svd(FOR_{r});
    [~, INV_S{r}, ~] = svd(INV_{r});
end
%% Plot Singular Value Results

figure(1)
clf
tiledlayout(2,4)
nexttile
semilogy(diag(SA))
xlabel('Singular Values')
title('A')

nexttile
semilogy(diag(SAdag))
xlabel('Singular Values')
title('pinv(A)')

nexttile
semilogy(sort(1./diag(SA),'descend'))
xlabel('Singular Values')
title('A-1')

nexttile
semilogy(diag(SB))
hold on 
semilogy(diag(SX))
legend('B','X')
xlabel('Singular Values')
title('Data')

nexttile
for r=1:N
    semilogy(diag(Mf_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('Mf')
xlim([0,N])

nexttile
for r=1:N
    semilogy(diag(Mi_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('Mi')
xlim([0,N])

nexttile
for r=1:N
    semilogy(diag(FOR_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('PAIR FORWARD')
semilogy(diag(SA),'k')

nexttile
for r=1:N
    semilogy(diag(INV_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('PAIR INVERSE')
semilogy(sort(1./diag(SA),'descend'),'k')
%% Larger Figure
figure(2)
for r=1:N
    semilogy(diag(FOR_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('PAIR FORWARD')
semilogy(diag(SA),'k')
%% Visualizing INVERSE and FORWARD results
tt = linspace(0,1,N);
idx = [25, 6, 35];

r = round(2*N/3);
figure(3)
clf
plot(tt,FOR_{r}*Xtrue(:,idx(1)), '--','Color', 'blue')
hold on
plot(tt,FOR_{r}*Xtrue(:,idx(2)), '--','Color', 'red')
plot(tt,FOR_{r}*Xtrue(:,idx(3)), '--','Color', 'green')
plot(tt,Btrue(:,idx(1)), '-','Color', 'blue')
plot(tt,Btrue(:,idx(2)), '-','Color', 'red')
plot(tt,Btrue(:,idx(3)), '-','Color', 'green')
box off
xlabel('time','Interpreter','latex')
ylabel('$x_{\rm pred}^{(j)}$','Interpreter','latex')
title('True and Predicted Observations','Interpreter','latex')
plot(0,0,'k-')
plot(0,0,'k--')
legend('','','','','','','True','Predicted','Interpreter','latex','Location','Northwest')

figure(4)
clf
plot(tt,INV_{r}*Bnoisy(:,idx(1)), '--','Color', 'blue')
hold on
plot(tt,INV_{r}*Bnoisy(:,idx(2)), '--','Color', 'red')
plot(tt,INV_{r}*Bnoisy(:,idx(3)), '--','Color', 'green')
plot(tt,Xtrue(:,25),'-','Color', 'blue')
plot(tt,Xtrue(:,6),'-','Color', 'red')
plot(tt,Xtrue(:,35),'-','Color', 'green')
box off
xlabel('time','Interpreter','latex')
ylabel('$b_{\rm pred}^{(j)}$','Interpreter','latex')
plot(0,0,'k-')
plot(0,0,'k--')
legend('','','','','','','True','Predicted','Interpreter','latex')
title('True and Predicted Parameters','Interpreter','latex')




