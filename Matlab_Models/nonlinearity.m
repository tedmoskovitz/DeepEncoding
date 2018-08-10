function [z] = nonlinearity(celltype, model, bound);

% get lnp model
initpaths;
pp = load(strcat("saved/",celltype, "_", model,"_model_2f.mat"));

% get filters
fs = csvread(strcat(celltype,"_2f.csv"));
f1 = fs(:,1);
f2 = fs(:,2); 

%f1 = csvread(strcat("/Users/TedMoskovitz/f1_", celltype,".csv"));
%f2 = csvread(strcat("/Users/TedMoskovitz/f2_", celltype,".csv"));

% regress 
g1 = reshape(pp.k(:,:,1), [],1); 
g2 = reshape(pp.k(:,:,2), [],1); 
w1 = [g1 g2] \ f1; 
w2 = [g1 g2] \ f2; 
% new filters
h1 = [g1 g2]*w1;  
h2 = [g1 g2]*w2;
% Convert to orthonormal basis via gram-schmidt orthogonalization
ufilts = gsorth([h1 h2]); % orthonormal basis for filter subspace
h1o = ufilts(:,1); % normalized filter 1
h2o = ufilts(:,2); % normalized orthogonalized filter 2

delta = 0.01;
xgrid = (-bound:delta:bound-delta); 
ygrid = (-bound:delta:bound-delta); 

npts = size(xgrid,2); 
z = zeros(npts,npts); 

% Get 2D grid of points at which to evaluate nonlinearity
[xx,yy] = meshgrid(xgrid,ygrid);

count = 0; 
for ix = 1:npts
    for iy = 1:npts
        if mod(count,10000) == 0
            disp(count); 
        end
        stm = xx(ix,iy)*h1o' + yy(ix,iy)*h2o'; 
        filtresponse = stm*[h1 h2];
        z(ix,iy) = pp.nlfun(filtresponse); 
        count = count + 1; 
    end
end

end