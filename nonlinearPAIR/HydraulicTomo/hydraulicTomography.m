% (c) Tia Chung December 2022
% matlab default model equation  
% m u'' + d u' - \nabla .(c \nabla u) + a u = f
%         d u' - \nabla .(c \nabla u) + a u = f
%
% nonlinear hydraulic tomography model equations
%    u - hydraulic head
%   \nabla (y(xi) . \nabla u) =  f
%    m = 0 (no second order time derivative)
%    d = 0 (time evolving)
%    c = -y (neg. hydraulic conductivity)
%    a = 0 (reaction term)
%    f = qi delta(xi_well)
%    qi - pumping rate
clear
clc
%% generate condactivity field function
dim = 5000;
XX = 2*(rand(dim,1)-0.5); YY = 2*(rand(dim,1)-0.5);
% 
% [X, Y] = meshgrid(linspace(xbegin, xend, nx), linspace(ybegin, yend, ny));
% 
% ZZ = conductivityPhantom(XX,YY);
% scatter3(XX,YY,log10(ZZ))
% xlabel('$x$')
% ylabel('$y$')
% zlabel('conductivity $K(x,y)$ [log$_{10}$]')

figure
xbegin = -1; xend = 1; ybegin = -1; yend = 1;    
nx = 100; ny = 100; % problem resolution
[X, Y] = meshgrid(linspace(xbegin, xend, nx), linspace(ybegin, yend, ny));
Z = -exp(-10*(X.^2 + Y.^2));
ZZ = conductivityField(XX, YY, X, Y, Z);
scatter3(XX,YY,log10(ZZ))

imshow(log(Z),[])
xlabel('$x$')
ylabel('$y$')
zlabel('conductivity $K(x,y)$ [log$_{10}$]')
%% generate data measurements
pumpingRate = 2; %  pumping rate in L/s
n_xwells = 4; n_ywells = 5;
well_x_location = linspace(-1,1,n_xwells+2); well_x_location = well_x_location(2:end-1);
well_y_location = linspace(-1,1,n_ywells+2); well_y_location = well_y_location(2:end-1);

count = 1;
for i = 1:n_xwells 
    for  j = 1:n_ywells 
        plot(well_x_location(i),well_y_location(j),'.','MarkerSize',15)
        well(count,:) = [well_x_location(i), well_y_location(j), pumpingRate];
        count = count + 1;
    end
end

%% generate pde
close all
pdeModel = createpde();                                                    % create empty scalar pde model
xbegin = -1; xend = 1; ybegin = -1; yend = 1;                              % create boundaries (rectangle)
R1 = [3, 4, xbegin, xend, xend, xbegin, ybegin, ybegin, yend, yend]';      % define rectangle geometry 
g = decsg(R1);                                                             % decompose constructive solid geometry into minimal regions
geometryFromEdges(pdeModel, g);                                            % include boundary geometri in pde model
applyBoundaryCondition(pdeModel, 'dirichlet', 'Edge', [1,2,3,4], 'u', 0);  % create boundary conditions
% applyBoundaryCondition(pdeModel, 'neumann', 'Edge', 3, 'u', 0);          % create boundary conditions
generateMesh(pdeModel,'Hmax',0.025); % create finite element mesh
figure, pdemesh(pdeModel)




for j = 1:length(well)
    xwell = well(j,1); ywell = well(j,2); rate = well(j,3);
    wellTerm  = @(l, s) exp(-150*(l.x - xwell).^2 -150*(l.y - ywell).^2);    % define forcing term
    f = @(l, s) wellTerm(l, s);                                            % right hand side of pde
%     z = @(l, s) -conductivityPhantom(l.x,l.y)';
    Z = -exp(-10*(X.^2 + Y.^2));
    z = @(l, s) conductivityField(l.x,l.y, X, Y, Z);
    specifyCoefficients(pdeModel,'m', 0, 'd', 0, 'c', z, 'a', 0, 'f', f);  % pde equation
    tic, u{j} = solvepde(pdeModel); toc                                       % solve pde model
    figure, pdeplot(pdeModel,'XYData',u{j}.NodalSolution(:,1)), hold on 
    plot(well(j,1),well(j,2),'.','MarkerSize',15)
    uData(j,:) = interpolateSolution(u{j},well([1:j-1,j+1:end],1),well([1:j-1,j+1:end],2));
    plot(well([1:j-1,j+1:end],1),well([1:j-1,j+1:end],2),'.','MarkerSize',15)
    drawnow 
end

%% test well function

wellModel2Data(Z, pdeModel, well, length(well))

%% generate noisy data

noiseLevel = 1;
btrue = uData(:); n_b = length(btrue);
% b = uData + noiseLevel*randn(length(well),);
%% plotting 
figure, pdeplot(pdeModel,'XYData',u.NodalSolution(:,1)), drawnow    % plot initial pde solution


plot(b)
% for j = 2:size(u.NodalSolution,2)                                          % plot initial pde solution
%     pdeplot(pdeModel,'XYData',u.NodalSolution(:,j)), drawnow
% end

%% initial condition function
% function u0 = initialCondition(forcingTerm, location, t0)
% s.time = t0;
% u0 = forcingTerm(location,s);
% end
function zInter = conductivityField(x,y, X, Y, Z)
zInter = interp2(X,Y,Z,x,y);
end