function [filts] = get_filters(celltype, model);

% get lnp model
initpaths;
pp = load(strcat("saved/",celltype, "_", model,"_model.mat"));

% get filters
filts = pp.k;
end