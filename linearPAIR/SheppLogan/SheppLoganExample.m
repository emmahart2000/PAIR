
% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%% 
% %
% % Linear PAIR Example: 64 x 64 Randomized Shepp Logan
% %
% %%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
% Creates linear PAIR for a toy CT imaging problem.  Generates figure 
% comparing method accuracies.  Option to generate images of diffent rank
% input/target representations.
% %%% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

%% Create or Load Phantoms
tic
if isfile('SheppLogan64data.mat')
    load('SheppLogan64data.mat')
else
    rng(0)                           % set seed
    n = 64;                          % image size
    nb = 12000;                      % number of independent sinograms
    nx = 10000;                       % number of independent phantoms
    nsup = 10000;                     % number of sino/phants for supervised task
    ntest = 2000;                    % number of paired sino/phants testing
    nimgs = nx+nb+nsup+ntest;      % number of images (x and b)

    % Create Shepp Logan Phantoms
    X = randomSheppLogan(n,{'pad', 0; 'M', nimgs});
    
    % Define Forward Operator A
    angles = linspace(1,180,36);          % define tomo angles
    p = 90;                               % define sinogram height
    options.angles = angles;              % set angles
    options.p = p;                        % set sinogram height
    A = PRtomo(n,options);                % construct A from PRtomo
    save('SheppLogan64A.mat', 'A')
    
    % Create Sinograms from Phantoms and Forward Operator, Add Noise
    noiseLevel = 0.05 ;                    % noise level
    Btrue = zeros(3240, nb+nsup+ntest);    % preallocate Btrue
    B = zeros(3240,nb+nsup+ntest);         % preallocate B   
    for j = nx+1:nimgs
        Btrue(:,j-nx) = A*X(:,j);            % create sinogram from phantom
        B(:,j-nx) = Btrue(:,j-nx) + WhiteNoise(Btrue(:,j-nx),noiseLevel);
    end

    % Divide into Testing and Training, Normalize
    Btrain = B(:,1:nb+nsup);              % split training
    Bmin = min(Btrain(:));                % normalize             
    Bmax = max(Btrain(:));
    B = (B - Bmin)/(Bmax-Bmin);
    save('SheppLogan64data.mat','X','B','Btrue','Bmin','Bmax')
end

%% Define Test Space and Training/Testing
r_xs = [1,2,3,4,5,100,500,1000,2000,3000];              % latent dimensions for targets
r_bs = [1,2,3,4,5,100,500,1000,2000,3000];              % latent dimensions for inputs
% r_xs = 50:50:3200;              % latent dimensions for targets
% r_bs = 50:50:3200;              % latent dimensions for inputs
% r_xs = 20:20:3240;              % latent dimensions for targets
% r_bs = 20:20:3240;              % latent dimensions for inputs

% X is stored like (Just X)(Just B)(X and B Pairs)(Testing X and B Pairs)
Xtrain_uns = X(:,1:nx);               % targets for unsupervised task (AEs)
Xtrain_sup = X(:,nx+nb+1:nx+nb+nsup); % targets for supervised task (latent maps)
Xtest = X(:,nimgs-ntest+1:nimgs);     % targets for testing

Btrain_uns = B(:,1:nb);               % inputs for unsupervised task (AEs)
Btrain_sup = B(:,nb+1:nb+nsup);       % inputs for supervised task (latent maps)
Btest = B(:,nb+nsup+1:nb+nsup+ntest); % inputs for testing

printimgs = 1;                        % printing option

%% Construct and Evaluate Linear PAIR for Different Latent Dimensions

if isfile('SheppLoganLinearPAIRerrors.mat')
    load('SheppLoganLinearPAIRerrors.mat')
else
    % Create Linear Autoencoders for Largest Latent Dimension
    [EX,DX,SX] = getAutoencoder(Xtrain_uns,max(r_xs));
    [EB,DB,SB] = getAutoencoder(Btrain_uns,max(r_bs));

    % Construct PAIR for Each r_x, r_b from Largest
    for i=1:length(r_xs)
        r_x = r_xs(i);          % latent dimension of x
        r_b = r_bs(i);          % latent dimension of b
    
        E_b = EB(1:r_b,:);      % b encoder (rank r_b) 
        D_b = DB(:,1:r_b);      % b decoder (rank r_b)
        E_x = EX(1:r_x,:);      % x encoder (rank r_x)
        D_x = DX(:,1:r_x);      % x decoder (rank r_x)

        % projected data
        Z_x = E_x*Xtrain_sup;   % latent x
        Z_b = E_b*Btrain_sup;   % latent b
        M_i = Z_x*pinv(Z_b);    % latent inverse map
        M_f = Z_b*pinv(Z_x);    % latent forward map

        % print images
        if printimgs == 1
            imwrite(squeeze(reshape(D_b*E_b*Btest(:,2),[90,36])), append('images/inputAE',string(r_x),'_',string(r_b),'.png'))
            imwrite(squeeze(reshape(D_x*E_x*Xtest(:,2),[64,64])), append('images/targetAE',string(r_x),'_',string(r_b),'.png'))
            imwrite(squeeze(reshape(D_x*M_i*E_b*Btest(:,2),[64,64])), append('images/PAIRinversion',string(r_x),'_',string(r_b),'.png'))
            imwrite(squeeze(reshape(D_b*M_f*E_x*Xtest(:,2),[90,36])), append('images/PAIRforward',string(r_x),'_',string(r_b),'.png'))
        end

    % errors
    BAEerr(i)     = avg_rel_error(D_b*E_b*Btest,Btest);
    XAEerr(i)     = avg_rel_error(D_x*E_x*Xtest,Xtest);
    PAIRinverr(i) = avg_rel_error(D_x*M_i*E_b*Btest,Xtest);
    PAIRforerr(i) = avg_rel_error(D_b*M_f*E_x*Xtest,Btest);

    end
    save('SheppLoganLinearPAIRerrors.mat', 'BAEerr', 'XAEerr', 'PAIRinverr', 'PAIRforerr')
end

%% Compare TSVD Inverse and Forward Approximation
if isfile('SheppLogan64TSVDerror.mat')
    load('SheppLogan64TSVDerror.mat')
else
    if isfile('SheppLogan64ASVD.mat')
        load('SheppLogan64ASVD.mat')
    else
        load('SheppLogan64A.mat')
        [U,S,V] = svd(full(A));    % compute SVD of forward map A
        save('SheppLoganASVD.mat','U','S','V')
    end
    B_test = Btest*(Bmax-Bmin) + Bmin; % unnormalize            
    ranks =  50:50:3200;            % define test space for TSVD
    %     ranks = 20:20:3240;
    for j = 1:length(ranks)
        r = ranks(j);
        Ur = U(:,1:r);             % take first r left singular vectors
        Vr = V(:,1:r);             % take first r right singular vectors
        Srinv = diag(1./diag(S(1:r,1:r))); % invert diagonal matrix
        Sr= diag(diag(S(1:r,1:r)));
        Xhat = Vr*(Srinv*(Ur'*B_test));
        Bhat = Ur*(Sr*(Vr'*Xtest));
        TSVDinverr(j) = avg_rel_error(Xhat,Xtest);
        TSVDforerr(j) = avg_rel_error(Bhat,B_test);
        disp(r)
    end
save('SheppLogan64TSVDerror.mat','TSVDinverr','TSVDforerr')
end

%% Graph Results
figure(1)
clf
set(0, 'DefaultAxesFontName', 'CMU Serif');
set(0, 'DefaultTextFontName', 'CMU Serif');
plot(ranks,XAEerr, '-', 'LineWidth',1.5)
hold on
plot(ranks,PAIRforerr, ':', 'LineWidth',1.5)
plot(ranks,TSVDforerr,'-*', 'LineWidth',1.5,'MarkerSize',3)
plot(ranks,BAEerr, '--', 'LineWidth',1.5,'MarkerSize',3)
plot(ranks,PAIRinverr, '-.', 'LineWidth',1.5)
plot(ranks,TSVDinverr,'-o', 'LineWidth',1.5,'MarkerSize',2)
legend('X Autoencoder', 'PAIR Forward', 'TSVD Forward', 'B Autoencoder', 'PAIR Inverse', 'TSVD Inverse', 'Location', 'NW','NumColumns',2)
xlabel('Rank')
ylabel('Average Relative Error')
xlim([0,3240])
ylim([0,1])
toc
