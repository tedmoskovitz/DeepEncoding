function [yhat] = lnp_predict(celltype, model, data_path);
% returns the predictions of a saved LNP model

% prep
initpaths;
load(strcat("../",data_path)); 
pp = load(strcat("saved/",celltype, "_", model,"_model.mat"));

% prediction
[logli,yhat] = logli_LNP(pp, stim, mean(spikes_per_frm,2)); % test log-likelihood

dtStim = .01; 
yhat = yhat ./ dtStim; 

end