% demo5_5spacetimefilters.m
%
% Tutorial script illustrating maximum likelihood / maximally informative
% dimensions (MID) estimation for LNP model with FIVE spatio-temporal
% filters.  Similar to demo3 but for space-time filters.
%
% - computes iSTAC estimate of five-filter model
% - computes ML estimate of five filter model with CBF nonlinearity
% - computes ML estimate of three-filter model with RBF nonlinearity (RBF not tractable for 5 dimensions).



% initialize paths
clear;
clf; 
initpaths;
%%
celltype = 'complex';
save_performance = false;
save_models = true; 
data_path = strcat('../RustV1/', celltype, '/data/');
rpt_path = strcat('../RustV1/', celltype, '/repeats/');
data_dir = dir(data_path);
rpt_dir = dir(rpt_path);
n_cells = numel(data_dir)-2;

nfilts_istac = 5; % number of iSTAC filters to compute
perf_istac_r2 = zeros(nfilts_istac, 1, n_cells); 
perf_istac_bps = zeros(nfilts_istac, 1, n_cells); 

% number of basis functions for CBF, RBF
% n_funcs = [3,5,10];
n_funcs = [3]; 
num_basis_funcs = size(n_funcs,2);

nfilts_cbf = 5; % number of CBF filters to compute
perf_cbf_r2 = zeros(nfilts_cbf, num_basis_funcs, n_cells); 
perf_cbf_bps = zeros(nfilts_cbf, num_basis_funcs, n_cells); 

nfilts_rbf = 3; % number of RBF filters to compute; max supported is 3
perf_rbf_r2 = zeros(nfilts_rbf, num_basis_funcs, n_cells); 
perf_rbf_bps = zeros(nfilts_rbf, num_basis_funcs, n_cells);

istac_shape = [nfilts_istac, 1, n_cells]; 
cbf_shape = [nfilts_cbf, num_basis_funcs, n_cells];
rbf_shape = [nfilts_rbf, num_basis_funcs, n_cells];


% load from checkpoint
load_frm_chkpt = false;
if load_frm_chkpt == true
    perf_istac_r2 = reshape(csvread(strcat('../SavedResults/',celltype,'_istac_r2.csv')), istac_shape);
    perf_istac_bps = reshape(csvread(strcat('../SavedResults/',celltype,'_istac_bps.csv')), istac_shape);
    perf_cbf_r2 = reshape(csvread(strcat('../SavedResults/',celltype,'_cbf_r2.csv')), cbf_shape);
    perf_cbf_bps = reshape(csvread(strcat('../SavedResults/',celltype,'_cbf_bps.csv')), cbf_shape);
    perf_rbf_r2 = reshape(csvread(strcat('../SavedResults/',celltype,'_rbf_r2.csv')), rbf_shape);
    perf_rbf_bps = reshape(csvread(strcat('../SavedResults/',celltype,'_rbf_bps.csv')), rbf_shape);
end


for c = 4:4%1:n_cells

    disp(strcat("Starting Cell ", string(c), " of ", string(n_cells)));
    
    % get training/test data
    load(strcat(data_path,data_dir(c+2).name));
    
    trainfrac = .8;
    ln = size(stim,1);
    Stim_tr = stim(1:round(trainfrac * ln),:);
    Stim_tst = stim(round(trainfrac*ln):end,:);
    sps_tr = spikes_per_frm(:,1:round(trainfrac * ln))';
    %sps_tr(sps_tr > 1) = 1;
    sps_tst = spikes_per_frm(:,round(trainfrac*ln):end)';
    %sps_tst(sps_tst > 1) = 1;
    RefreshRate = .01;
    width = size(Stim_tr,1); % simple 16 complex 24
    
    slen_tr = size(Stim_tr,1);   % length of training stimulus / spike train
    slen_tst = size(Stim_tst,1); % length of test stimulus / spike train
    nkx = size(Stim_tr,2); % number of spatial pixels in stimulus
    nsp_tr = sum(sps_tr);   % number of spikes in training set
    nsp_tst = sum(sps_tst); % number of spikes in test set
    
    % get repeat data
%     load(strcat(rpt_path,rpt_dir(c+2).name));
%     Stim_rpt = stim;
%     


    % set up functionality for BPS calculations
    % Compute log-likelihood of constant rate (homogeneous Poisson) model
    muspike_tr = nsp_tr/slen_tr;    % mean number of spikes / bin, training set
    muspike_tst = nsp_tst/slen_tst; % mean number of spikes / bin, test set
    LL0_tr =   nsp_tr*log(muspike_tr) - slen_tr*muspike_tr; % log-likelihood, training data
    LL0_tst = nsp_tst*log(muspike_tst) - slen_tst*muspike_tst; % log-likelihood test data
    
    % Functions to compute single-spike informations
    f1 = @(x)((x-LL0_tr)/nsp_tr/log(2)); % compute training single-spike info
    f2 = @(x)((x-LL0_tst)/nsp_tst/log(2)); % compute test single-spike info
    % (Divide by log 2 to get 'bits' instead of 'nats')
    

    %% == 2. Compute iSTAC estimator (for comparison sake)
    
    nkt = 16; % number of time bins to use for filter
    % This is important: normally would want to vary this to find optimal filter length
    
    % Compute STA and STC
    [sta,stc,rawmu,rawcov] = simpleSTC(Stim_tr,sps_tr,nkt);  % compute STA and STC
    
    % Compute iSTAC estimator
    fprintf('\nComputing iSTAC estimate\n');
    [istacFilts,vals,DD] = compiSTAC(sta(:),stc,rawmu,rawcov,nfilts_istac); % find iSTAC filters
    
    
    
    % Compute training and test log-likelihood for each sized model
    LListac_tr = zeros(nfilts_istac,1);
    LListac_tst = zeros(nfilts_istac,1);
    istac_r2s = zeros(nfilts_istac,1);
    for jj = 1:nfilts_istac
        
        if perf_istac_bps(jj,1,c) == 0 % don't overwrite 
            % Fit iSTAC model nonlinearity using varying # of filters
            pp_istac = fitNlin_expquad_ML(Stim_tr,sps_tr,istacFilts(:,1:jj),RefreshRate); % LNP model struct
            % compute train and test log-likelihood
            LListac_tr(jj) = logli_LNP(pp_istac,Stim_tr,sps_tr); % training log-likelihood
            [LListac_tst(jj),rate_istac] = logli_LNP(pp_istac,Stim_tst,sps_tst); % test log-likelihood
            r = corr2(rate_istac, sps_tst);
            istac_r2s(jj) = r * r;
            
            perf_istac_r2(jj,1,c) = r*r;
            
            perf_istac_bps(jj,1,c) = f2(LListac_tst(jj));
        end
    end
    istac_r = corr2(rate_istac, sps_tst);
    istac_r2 = istac_r * istac_r; % R2

    
    % save filters
%     filt_dir = "/Users/TedMoskovitz/Thesis/Models/NeuralNets/V1/LNP_filters/complex1";
%     filt_dir = filt_dir + "/5f_";
%     
%     rfilt = reshape(istacFilts(:,1:nfilts_istac), [nfilts_istac,width*nkt]);
%     %csvwrite(filt_dir + "istac.csv", rfilt);
%     
%    csvwrite(strcat('../SavedFilters/',celltype,'_istac_filters_cell',string(c),'.csv'), istacFilts);
%     %% -----  Plot filters ------------
%     rs = @(x)(reshape(x,nkt,nkx)); % reshape as image
%     
%     clf;
%     subplot(231);
%     tt = -nkt+1:0; xx = 1:nkx;
%     
    %% == 3. Set up temporal basis for stimulus filters (for ML / MID estimators)
    
    % Set up fitting structure and compute initial logli
    mask = [];  % time range to use for fitting (set to [] if not needed).
    pp0 = makeFittingStruct_LNP(sta,RefreshRate,mask); % initialize param struct
    
    % == Set up temporal basis for representing filters  ====
    % (try changing these params until basis can accurately represent STA).
    ktbasprs.neye = 0; % number of "identity"-like basis vectors
    ktbasprs.ncos = 6; % number of raised cosine basis vectors (DETERMINES BASIS DIMENSIONALITY)
    ktbasprs.kpeaks = [0 3*nkt/4]; % location of 1st and last basis vector bump
    ktbasprs.b = 50; % determines how nonlinearly to stretch basis (higher => more linear)
    [ktbas, ktbasis] = makeBasis_StimKernel(ktbasprs, nkt); % make basis
    filtprs_basis = (ktbas'*ktbas)\(ktbas'*sta);  % filter represented in new basis
    sta_basis = ktbas*filtprs_basis;
    
    % Insert filter basis into fitting struct
    pp0.k = sta_basis; % insert sta filter
    pp0.kt = filtprs_basis; % filter coefficients (in temporal basis)
    pp0.ktbas = ktbas; % temporal basis
    pp0.ktbasprs = ktbasprs;  % parameters that define the temporal basis
    
    %% == 4. ML / MID estimator for LNP with CBF (cylindrical basis func) nonlinearity
    
    for nf = 1:num_basis_funcs
        % Set parameters for cylindrical basis funcs (CBFs) and initialize fit
        fprintf("Fitting models with %d basis functions on Cell %d...", n_funcs(1,nf), c); 
        fstructCBF.nfuncs = n_funcs(1,nf); % number of basis functions for nonlinearity
        fstructCBF.epprob = [.01, 0.99]; % cumulative probability outside outermost basis function peaks
        fstructCBF.nloutfun = @logexp1;  % log(1+exp(x)) % nonlinear stretching function
        fstructCBF.nlfuntype = 'cbf';
        
        % Fit the model (iteratively adding one filter at a time)
        optArgs = {'display','iter'};
        [ppcbf,negLcbf,ppcbf_array] = fitLNP_multifiltsCBF_ML(pp0,Stim_tr,sps_tr,nfilts_cbf,fstructCBF,istacFilts,optArgs);
        
        % Compute training and test log-likelihood for each sized model
        LLcbf_tr = zeros(nfilts_cbf,1);
        LLcbf_tst = zeros(nfilts_cbf,1);
        cbf_r2s = zeros(nfilts_cbf,1);
        for jj = 1:nfilts_cbf
            
            if perf_cbf_bps(jj,nf,c) == 0
                
                % compute train and test log-likelihood
                LLcbf_tr(jj) = logli_LNP(ppcbf_array{jj},Stim_tr,sps_tr); % training log-likelihood
                [LLcbf_tst(jj),rate_cbf] = logli_LNP(ppcbf_array{jj},Stim_tst,sps_tst); % test log-likelihood
                r = corr2(rate_cbf, sps_tst);
                cbf_r2s(jj) = r * r;
                
                perf_cbf_r2(jj,nf,c) = r*r;
                
                perf_cbf_bps(jj,nf,c) = f2(LLcbf_tst(jj));
                
            end
        end
        
        % R-Squared
        cbf_r = corr2(rate_cbf, sps_tst);
        cbf_r2 = cbf_r * cbf_r;
        
        % save filters
        filt_dir = "/Users/TedMoskovitz/Thesis/Models/NeuralNets/V1/LNP_filters/complex1";
        filt_dir = filt_dir + "/5f_";
        
        filts_cbf = reshape(ppcbf_array{end}.k,nkt*nkx,nfilts_cbf);
        
        
        rfilt = reshape(filts_cbf(:,1:nfilts_cbf), [nfilts_cbf,nkx*nkt]);
        %csvwrite(filt_dir + "cbf.csv", rfilt);
        
        
        
        % Determine which of these models is best based on xv log-likelihood
        [~,imax_cbf] = max(LLcbf_tst);
        fprintf('LNP-CBF: best performance for model with %d filters\n',imax_cbf);
        %ppcbf= ppcbf_array{imax_cbf}; % select this model
        ppcbf = ppcbf_array{5};
        
        % Compute true filters reconstructed in basis of CBF filter estimates
        % (using 5-filter model)
        filts_cbf = reshape(ppcbf_array{end}.k,nkt*nkx,nfilts_cbf);  % filter estimates
    end
    
    if save_performance == true
        disp("Saving Checkpoint...");
        csvwrite(strcat('../SavedResults/',celltype,'_istac_r2.csv'), reshape(perf_istac_r2, [],1));
        csvwrite(strcat('../SavedResults/',celltype,'_istac_bps.csv'), reshape(perf_istac_bps, [],1));
        csvwrite(strcat('../SavedResults/',celltype,'_cbf_r2.csv'), reshape(perf_cbf_r2, [],1));
        csvwrite(strcat('../SavedResults/',celltype,'_cbf_bps.csv'), perf_cbf_bps, [],1);
    end
% 
%     %% == 5. ML / MID 2:  ML estimator for LNP with RBF (radial basis func) nonlinearity
% 
%     for nf = 1:num_basis_funcs
%         % Set parameters for cylindrical basis funcs (CBFs) and initialize fit
%         fprintf("Fitting models with %d basis functions on Cell %d...", n_funcs(1,nf), c); 
%         fstructRBF.nfuncs = n_funcs(1,nf); % number of basis functions for nonlinearity
%         fstructRBF.epprob = [.01, 0.99]; % cumulative probability outside outermost basis function peaks
%         fstructRBF.nloutfun = @logexp1;  % log(1+exp(x)) % nonlinear stretching function
%         fstructRBF.nlfuntype = 'rbf';
%         
%         % Fit the model (iteratively adding one filter at a time)
%         [pprbf,negLrbf,pprbf_array] = fitLNP_multifiltsRBF_ML(pp0,Stim_tr,sps_tr,nfilts_rbf,fstructRBF,istacFilts);
%         
%         % Compute training and test log-likelihood for each sized model
%         LLrbf_tr = zeros(nfilts_rbf,1);
%         LLrbf_tst = zeros(nfilts_rbf,1);
%         rbf_r2s = zeros(nfilts_rbf,1);
%         for jj = 1:nfilts_rbf
%             
%             if perf_rbf_bps(jj,nf,c) == 0
%                 
%                 % compute train and test log-likelihood
%                 LLrbf_tr(jj) = logli_LNP(pprbf_array{jj},Stim_tr,sps_tr); % training log-likelihood
%                 [LLrbf_tst(jj),rate_rbf] = logli_LNP(pprbf_array{jj},Stim_tst,sps_tst); % test log-likelihood
%                 r = corr2(rate_rbf, sps_tst);
%                 rbf_r2s(jj) = r * r;
%                 
%                 perf_rbf_r2(jj,nf,c) = r * r;
%                 
%                 perf_rbf_bps(jj,nf,c) = f2(LLrbf_tst(jj));
%                 
%             end
%         end
%         
%         % R-squared
%         rbf_r = corr2(rate_rbf, sps_tst);
%         rbf_r2 = rbf_r * rbf_r;
%         
%         % Determine which of these models is best based on xv log-likelihood
%         [~,imax_rbf] = max(LLrbf_tst);
%         fprintf('LNP-RBF: best performance for model with %d filters\n',imax_rbf);
%         %pprbf = pprbf_array{imax_rbf}; % select this model
%         pprbf = pprbf_array{3};
%     
%     end
% %     %Compute true filters reconstructed in basis of RBF filter estimates
% %     %
% %     filts_rbf = reshape(pprbf.k,nkt*nkx,size(pprbf.k,3));  % filter estimates
% % %     %
% %     REPEAT PREDICTION
% %     stim_rpt = csvread('/Users/TedMoskovitz/Thesis/Models/NeuralNets/Data/RustV1dat/X_rpt_stim_c.csv');
% %     %sps_rpt = csvread('/Users/TedMoskovitz/Thesis/Models/NeuralNets/Data/RGC_data/y_rpt_stim6.csv');
% %     [LLrbf_tst,rate_rbf_rpt, th] = logli_LNP(pprbf_array{nfilts_rbf},stim_rpt,stim_rpt(:,1));
% %     %csvwrite('/Users/TedMoskovitz/Thesis/Models/NeuralNets/V1/performance/rpt_preds_c_rbf.csv', rate_rbf_rpt);
% %     
%     
% %     % save filters
% %     filt_dir = "/Users/TedMoskovitz/Thesis/Models/NeuralNets/V1/LNP_filters/complex1";
% %     filt_dir = filt_dir + "/3f_";
% %     %
% %     filts_rbf = reshape(pprbf_array{end}.k,nkt*nkx,nfilts_rbf);
% %     
% %     
% %     rfilt = reshape(filts_rbf(:,1:nfilts_rbf), [nfilts_rbf,nkx*nkt]);
%     %csvwrite(filt_dir + "rbf.csv", rfilt);
%     
%     % 6. ==== Compute training and test performance in bits/spike =====
%     %%
%     %==== report R2 error in reconstructing first two filters =====
% %     kmse = sum((filts_true(:)-mean(filts_true(:))).^2); % mse of these two filters around mean
% %     ferr = @(k)(sum((filts_true(:)-k(:)).^2)); % error in optimal R2 reconstruction of ktrue
% %     fRsq = @(k)(1-ferr(k)/kmse); % R-squared
% % %     
% %     Rsqvals = [fRsq(filtsHat_istac),fRsq(filtsHat_cbf),.01];%fRsq(filtsHat_rbf)];
% % %     
% %     fprintf('\n=========== RESULTS ======================================\n');
% %     fprintf('\nFilter R^2:\n------------\n');
% %     fprintf('istac:%.2f cbf:%.2f rbf:%.2f\n',Rsqvals);
% %     
% %     % Plot filter R^2 values
% %     subplot(5,5,21);
% %     axlabels = {'istac','cbf','rbf'};
% %     bar(Rsqvals); ylabel('R^2'); title('filter R^2');
% %     set(gca,'xticklabel',axlabels, 'ylim', [.9*min(Rsqvals) 1.1*max(Rsqvals)]);
% %     
% %     %
% %     PSTH --------------------------
% %     prefix = "/Users/TedMoskovitz/Thesis/Models/NeuralNets/Data/RustV1dat/";
% %     pred_prefix = "/Users/TedMoskovitz/Thesis/Models/NeuralNets/V1/PSTHs/";
% %     
% %     %
% %     xte = xte(:,-15:end);
% %     csvwrite(pred_prefix + "5f_istac_c1.csv", rate_istac(1:100));
% %     csvwrite(pred_prefix + "5f_cbf_c1.csv", rate_cbf(1:100));
% %     csvwrite(pred_prefix + "3f_rbf_c1.csv", rate_rbf(1:100));
% %     x = 1:100;
% %     subplot(235);
% %     stem(x,sps_tst(x)); hold on;
% %     plot(x, rate_istac(1:100)/RefreshRate);
% %     hold off;
% %     
% %     %
%     
%     
%     %====== Compute log-likelihood / single-spike information ==========
%     
%     % Compute log-likelihood of constant rate (homogeneous Poisson) model
%     muspike_tr = nsp_tr/slen_tr;    % mean number of spikes / bin, training set
%     muspike_tst = nsp_tst/slen_tst; % mean number of spikes / bin, test set
%     LL0_tr =   nsp_tr*log(muspike_tr) - slen_tr*muspike_tr; % log-likelihood, training data
%     LL0_tst = nsp_tst*log(muspike_tst) - slen_tst*muspike_tst; % log-likelihood test data
%     
%     % Functions to compute single-spike informations
%     f1 = @(x)((x-LL0_tr)/nsp_tr/log(2)); % compute training single-spike info
%     f2 = @(x)((x-LL0_tst)/nsp_tst/log(2)); % compute test single-spike info
%     % (Divide by log 2 to get 'bits' instead of 'nats')
%     
%     % Compute single-spike info for each model
%     SSinf_istac_tr = f1(LListac_tr);   % training data
%     SSinf_istac_tst = f2(LListac_tst); % test data
%     %SSinf_cbf_tr = f1(LLcbf_tr);   % training data
%     %SSinf_cbf_tst = f2(LLcbf_tst); % test data
%     SSinf_rbf_tr = f1(LLrbf_tr);   % training data
%     SSinf_rbf_tst = f2(LLrbf_tst); % test data
%     %SSr2_tst = [istac_r2, cbf_r2, rbf_r2];
%     
%     % save performance
% %     dir = "/Users/TedMoskovitz/Thesis/Models/NeuralNets/V1/performance/performance_c1_lnp_";
% %     csvwrite(dir + "istac.csv", SSinf_istac_tst);
% %     csvwrite(dir + "cbf.csv", SSinf_cbf_tst);
% %     csvwrite(dir + "rbf.csv", SSinf_rbf_tst);
% %     
% %     csvwrite(dir + "istac_r2.csv", istac_r2s);
% %     csvwrite(dir + "cbf_r2.csv", cbf_r2s);
% %     csvwrite(dir + "rbf_r2.csv", rbf_r2s);
% %     
% %     
% %     SSinfo_tr = [SSinf_istac_tr(end), SSinf_cbf_tr(end),SSinf_rbf_tr(end)];%SSinf_rbf_tr(end)];
% %     SSinfo_tst = [SSinf_istac_tst(end), SSinf_cbf_tst(end),SSinf_rbf_tst(end)];%SSinf_rbf_tst(end)];
% %     
% %     fprintf('\nSingle-spike information (bits/spike):\n');
% %     fprintf('------------------------------------- \n');
% %     fprintf('Train: istac: %.2f  cbf:%.2f  rbf:%.2f\n', SSinfo_tr);
% %     fprintf('Test:  istac: %.2f  cbf:%.2f  rbf:%.2f\n', SSinfo_tst);
% %     fprintf('R-2:   istac: %.2f  cbf:%.2f  rbf:%.2f\n', SSr2_tst);
% %     
%     
% %     DONE
% %     amp=10;
% %     fs=20500;  % sampling frequency
% %     duration=1;
% %     freq=100;
% %     values=0:1/fs:duration;
% %     a=amp*sin(2*pi* freq*values);
% %     sound(a)
% %     %Plot test single-spike information
% %     %Plot filter R^2 values
% %     subplot(5,5,21);
% %     axlabels = {'istac','cbf','rbf'};
% %     bar(Rsqvals); ylabel('R^2'); title('filter R^2');
% %     set(gca,'xticklabel',axlabels, 'ylim', [.9*min(Rsqvals) 1.1*max(Rsqvals)]);
% %     
% %     subplot(5,5,21:23);
% %     cols = get(gca,'colororder');
% %     h = plot(1:nfilts_istac,SSinf_istac_tst,'-o',...
% %         1:nfilts_cbf,SSinf_cbf_tst,'-o',...
% %         1:nfilts_rbf,SSinf_rbf_tst,'-o',...
% %         1:nfilts_istac,SSinf_istac_tr,'o--',...
% %         1:nfilts_cbf,SSinf_cbf_tr,'o--',...
% %         1:nfilts_rbf,SSinf_rbf_tr,'o--', 'linewidth', 2);
% %     set(h(4),'color',cols(1,:));
% %     set(h(5),'color',cols(2,:));
% %     set(h(6),'color',cols(3,:)); axis tight;
% %     ylabel('bits / sp'); title('test single spike info');
% %     xlabel('# filters'); legend('istac','cbf','rbf');
% %     title('train and test log-likelihood');
% %     
% %     %Plot some rate predictions for first 100 bins
% %     subplot(5,5,24:25);
% %     ii = 1:100;
% %     stem(ii, sps_tst(ii)); hold on;
% %     plot(ii,rate_istac(ii)/RefreshRate,ii,rate_cbf(ii)/RefreshRate,ii,rate_rbf(ii)/RefreshRate, 'linewidth',2);
% %     title('rate predictions on test data (3-filter models)');
% %     legend('istac', 'ml-cbf','ml-rbf');
% %     ylabel('rate (sp/s)'); xlabel('time bin');
% %     
%     
%     %%
% %     % nonlin experiment
% %     prefix = "/Users/TedMoskovitz/Thesis/Models/NeuralNets/V1/params/";
% %     %b1 = csvread(prefix + "complex_2filt1cpy_b11_rect.csv");
% %     %b2 = csvread(prefix + "complex_2filt1cpy_b21_rect.csv");
% %     f1 = csvread(strcat("/Users/TedMoskovitz/f1_", celltype,".csv"));
% %     f2 = csvread(strcat("/Users/TedMoskovitz/f2_", celltype,".csv"));
% %     %W2 = csvread(prefix + "complex_2filt1cpy_W21_rect.csv");
% %     
% %     
% %     % regress
% %     dim1 = size(ppcbf.k(:,:,1), 1);
% %     dim2 = size(ppcbf.k(:,:,1), 2);
% %     g1 = reshape(ppcbf.k(:,:,1), [dim1*dim2,1]);
% %     g2 = reshape(ppcbf.k(:,:,2), [dim1*dim2,1]);
% %     w1 = [g1 g2] \ f1; % [pprbf.k(:,:,1) \ f1, pprbf.k(:,:,2) \ f1];
% %     w2 = [g1 g2] \ f2; % [pprbf.k(:,:,1) \ f2, pprbf.k(:,:,2) \ f2];
% %     % new filters
% %     h1 = [g1 g2]*w1;
% %     h2 = [g1 g2]*w2;
% %     % Convert to orthonormal basis via gram-schmidt orthogonalization
% %     ufilts = gsorth([h1 h2]); % orthonormal basis for filter subspace
% %     h1o = ufilts(:,1); % normalized filter 1
% %     h2o = ufilts(:,2); % normalized orthogonalized filter 2
% %     
% %     lim = 2.5; % 3.0 for complex
% %     step = .01;
% %     xgrid = (-lim : step : lim-step);
% %     ygrid = (-lim : step : lim-step);
% %     
% %     npts = size(xgrid,2);
% %     z = zeros(npts,npts);
% %     
% %     % Get 2D grid of points at which to evaluate nonlinearity
% %     [xx,yy] = meshgrid(xgrid,ygrid);
% %     
% %     count = 0;
% %     for ix = 1:npts
% %         for iy = 1:npts
% %             if mod(count,10000) == 0
% %                 disp(count/(npts*npts));
% %             end
% %             stm = xx(ix,iy)*h1o' + yy(ix,iy)*h2o';
% %             filtresponse = stm*[h1 h2];
% %             z(ix,iy) = ppcbf.nlfun(filtresponse);
% %             count = count + 1;
% %         end
% %     end
% %     %
% %     clf;
% %     z_nn = csvread('/Users/TedMoskovitz/z_simple_nn.csv');
% %     subplot(244);  % Plot filters
% %     %plot(tt,squeeze([h1 h2]),'linewidth',2);
% %     axis tight; title('ML-NN filters');
% %     subplot(248); mesh(xgrid,ygrid,z_nn);
% %     hold on; mesh(xgrid, ygrid, z);
% %     axis tight; %set(gca,'zlim',2);
% %     xlabel('filter 1 axis');ylabel('filter 2 axis');
% %     zlabel('spike rate (sps/sec)'); title('2D nn nonlinearity');
% %     
% %     
%     %%
%     if save_performance == true
%         disp("Saving Checkpoint...");
%         csvwrite(strcat('../SavedResults/',celltype,'_istac_r2.csv'), reshape(perf_istac_r2, [],1));
%         csvwrite(strcat('../SavedResults/',celltype,'_istac_bps.csv'), reshape(perf_istac_bps, [],1));
%         csvwrite(strcat('../SavedResults/',celltype,'_cbf_r2.csv'), reshape(perf_cbf_r2, [],1));
%         csvwrite(strcat('../SavedResults/',celltype,'_cbf_bps.csv'), reshape(perf_cbf_bps, [],1));
%         csvwrite(strcat('../SavedResults/',celltype,'_rbf_r2.csv'), reshape(perf_rbf_r2, [],1));
%         csvwrite(strcat('../SavedResults/',celltype,'_rbf_bps.csv'), reshape(perf_rbf_bps, [],1));
%     end
%     
    if save_models == true
        disp("Saving Best Models...");
        %save(strcat("saved/",celltype,"_istac_model.mat"), '-struct', 'pp_istac')
        save(strcat("saved/",celltype,"_cbf_model.mat"), '-struct', 'ppcbf')
        %save(strcat("saved/",celltype,"_rbf_model.mat"), '-struct', 'pprbf')
    end
    
    
end