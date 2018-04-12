% initpaths.m
%
% Initialize paths needed for MIDlnp code

addpath tools_iSTAC   % directory for iSTAC (used to initialize filter estimates)
addpath tools_LNPfitting  % directory for MID code
addpath tools_simdata;    % directory for simulating data (for demo code)
addpath nlfuns;   % directory for nonlinearities
addpath(genpath(pwd)); % directory for minFunc gradient descent package


if ~exist('simdatadir','dir')
    mkdir simdatadir;
end
