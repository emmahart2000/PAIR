function d = wellModel2Data(Z, pdeModel, well, nwell)

% Z = reshape(z)
% nwell = size(well,1); nd = nwell*(nwell-1);
% d = zeros(size(well,1),size(well,1)-1)
d = zeros(nwell,nwell-1);
z = @(l, s) interp2(X,Y,Z,l.x,l.y);                                        % get conductivity field

for j = 1:length(well)                                         
    f = @(l, s) exp(-150*(l.x - well(j,1)).^2 -150*(l.y - well(j,2)).^2);  % right hand side of pde
    specifyCoefficients(pdeModel,'m', 0, 'd', 0, 'c', z, 'a', 0, 'f', f);  % pde equation
    u = solvepde(pdeModel);                                                % solve pde model
    d(j,:) = interpolateSolution(u,well([1:j-1,j+1:end],1),well([1:j-1,j+1:end],2)); % project model on data
end

% d = d(:);

end
