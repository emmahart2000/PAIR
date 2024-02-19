function [E,D,S,L] = getAutoencoder(X,r) % get a matrix square root via eigenvalue decomposition
[D,S,~] = svds(X*X',r); 
S = 1/size(X,2)*diag(S);
E = D';
if nargout > 3, L = D*diag(sqrt(S)); end
end
