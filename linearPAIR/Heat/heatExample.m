% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%% 
% %
% % Linear PAIR Example: 100 Heat Example
% %
% %%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
% Creates linear PAIR for a toy heat problem.  Calculates and compares 
% singular valules for different latent/full maps.  Visualizes PAIR 
% results on 1 dimensional example.
% %%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

%% Create Data
rng(21)

N = 100;
A = heat(N);
K = 1000;
 
t = linspace(0,1,N)'/2; Xtrue = zeros(N,K); Bnoisy = zeros(N,K); Btrue = zeros(N,K);
minNoise = 1e-4; maxNoise = 1e-2;
noiseLevel = minNoise + (maxNoise-minNoise)*rand(1,K);

for k = 1:K
      r = randi(10,3) - 0.5*rand(3,1);
      x = sin(r(1)*2*pi*t) + sin(r(2)*2*pi*t);
      Xtrue(:,k) = x + abs(min(x));
  Btrue(:,k) = A*Xtrue(:,k);
  Bnoisy(:,k) = Btrue(:,k) + noiseLevel(k)*randn(N,1);
end
Adagger = pinv(A);

%% Take Necessary SVDs
[UX, SX, VX] = svd(Xtrue);
[UB, SB, VB] = svd(Bnoisy);
[UA, SA, VA] = svd(A);
[UAdag, SAdag, VAdag] = svd(Adagger);

%% For Different Ranks, Construct AEs and PAIR Mappings 
for r=1:100
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
for r=1:100
    semilogy(diag(Mf_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('Mf')
xlim([0,100])

nexttile
for r=1:100
    semilogy(diag(Mi_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('Mi')
xlim([0,100])

nexttile
for r=1:100
    semilogy(diag(FOR_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('PAIR FORWARD')
semilogy(diag(SA),'k')

nexttile
for r=1:100
    semilogy(diag(INV_S{r}))
    hold on
end
xlabel('Singular Values for Varied r')
title('PAIR INVERSE')
semilogy(sort(1./diag(SA),'descend'),'k')

%% Visualizing INVERSE and FORWARD results
tt = linspace(0,1,100);
idx = [25, 6, 35];

r = 10;
figure(10)
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

figure(11)
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

