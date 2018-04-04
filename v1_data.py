# V1 data
# Ted Moskovitz, 2018

import numpy as np
import h5py
import scipy.io as spio
import os

np.random.seed(324)

class V1:

	def __init__(self, cell, cell_num, 
                 n_frames=16, spk_thresh=None, flatten=True,
                 shuff=True, gen_new=True, verbose=True):
		"""
		cell = 'simple' or 'complex'
		spk_thresh if want to threshold spike counts
		"""
		self.shuff = shuff
		self.cell = cell
		self.cell_num = cell_num
		self.flatten = flatten
		self.spk_thresh = spk_thresh
		self.n_cells = -1
		self.n_frames = n_frames
		if gen_new:
			if verbose: print ('building data...');
			self.generate_new()
			if verbose: print ('fetching repeat data...');
			self.gen_rpt()
		else: 
			print ('retrieving data...')
			print ('This action is unsupported at this time.')
		if verbose: print ('done.'); 

        
	def gen_rpt(self):
		path = 'RustV1/' + self.cell + '/repeats/'
		directory = sorted(os.listdir(path))
		self.n_cells = len(directory)
		assert(self.cell_num <= self.n_cells)
		self.rpt_file = path + directory[self.cell_num-1]
		rpt_data = h5py.File(path + directory[self.cell_num-1], 'r')
        
		spks = rpt_data['spk_tms']
		spks_per_frm = rpt_data['spikes_per_frm'] # nrpt x nfrm
		self.y_rpt_lnp = spks_per_frm # for lnp
		nrpt = spks_per_frm.shape[0]
		self.nrpt = nrpt
        
		# spike times
		self.Mtsp = []
		for i in range(nrpt):
			st = spks[i][0]
			obj = np.asarray(rpt_data[st])
			self.Mtsp.append(obj)
            
		# stimulus
		StimRpt = np.asarray(rpt_data['stim']).T
		self.X_rpt_lnp = rpt_data['stim'] # formatted for lnp
		self.framelen = .01 # 100Hz
		self.Stim_rpt = StimRpt # un-batched stim (aka for LNP model)
		self.rpt_nstim = len(StimRpt)
        
        
		X_rpt = np.vstack([np.zeros([self.n_frames,StimRpt.shape[1]]), StimRpt])

		width = X_rpt.shape[1]
		T = X_rpt.shape[0]
		binned_rpts = np.zeros((T-self.n_frames, self.n_frames, width))
		for i in range(self.n_frames, T):
			# if len(Stim[i+stim_bin : i]) >= stim_bin:
			if len(X_rpt[i-self.n_frames : i, :]) > (self.n_frames-1):
				binned_rpts[i-self.n_frames, :, :] = X_rpt[i-self.n_frames : i, :]

		binned_rpts -= np.mean(binned_rpts, axis=0)
		# flatten
		if self.flatten:
			binned_rpts = np.reshape(binned_rpts, (T-self.n_frames, self.n_frames*width))
		else:
			binned_rpts = np.reshape(binned_rpts, (T-self.n_frames, self.n_frames, width))
            
		self.X_rpt = binned_rpts
        
		self.gen_raster()
		self.get_psth()
		self.get_r2()
        
	def gen_raster(self):
		# bin spikes and compute PSTH
		dtbin = .001 # in seconds
		framelen = .01
		nsec = framelen * self.rpt_nstim
		Tsp = self.Mtsp
		binctrs = np.arange(dtbin/2., nsec + 3*dtbin/2., dtbin) 
		nbin = len(binctrs)-1
		Mraster = np.zeros((self.nrpt, nbin))
		for jj in range(self.nrpt):
			iisp = (Tsp[jj] > 0)
			h,_ = np.histogram(Tsp[jj]*framelen, binctrs - dtbin/2)
			Mraster[jj, :] = h
		self.rpt_raster = Mraster
        
	def get_r2(self):
		nrpt = self.Mraster.shape[0]
		half1 = self.Mraster[:int(self.nrpt/2),:]
		half2 = self.Mraster[int(self.nrpt/2):,:]
		avg_h1 = np.mean(half1, axis=0).reshape(-1,)
		avg_h2 = np.mean(half2, axis=0).reshape(-1,)
		self.r = np.corrcoef(avg_h1, avg_h2)[0,1]
		self.r2 = self.r ** 2.
        
	def get_psth(self):
		nstim = len(self.Stim_rpt)
		self.nstim = nstim
		nsec = nstim * self.framelen
        
		dtbin = .001 # in seconds
		# bin spikes and compute psth
		Tsp = self.Mtsp
		binctrs = np.arange(dtbin/2., nsec + 3*dtbin/2., dtbin) 
		nbin = len(binctrs)-1
		self.Mraster = np.zeros((self.nrpt, nbin))
		for jj in range(self.nrpt):
			h,_ = np.histogram(Tsp[jj]*self.framelen, binctrs - dtbin/2)
			self.Mraster[jj, :] = h
		self.psth = np.mean(self.Mraster, axis=0) / dtbin     
        

	def generate_new(self):
		path = './RustV1/' + self.cell + '/data/'
		directory  = sorted(os.listdir(path))
		self.n_cells = len(directory)
		assert(self.cell_num <= self.n_cells)
		data = h5py.File(path + directory[self.cell_num-1], 'r')


		Stim = np.asarray(data['stim'])
		spks_per_frm = np.asarray(data['spikes_per_frm'])

		Stim = Stim.T
		width = Stim.shape[1]
		spks_per_frm = np.reshape(spks_per_frm, (-1,))

		n_frames = self.n_frames
		T = len(spks_per_frm)
		binned_stims = np.zeros((T-n_frames, n_frames, width))
		for i in range(n_frames, Stim.shape[0]):
			# if len(Stim[i+stim_bin : i]) >= stim_bin:
			if len(Stim[i-n_frames : i, :]) > (n_frames-1):
				binned_stims[i-n_frames, :, :] = Stim[i-n_frames : i, :]

		binned_stims -= np.mean(binned_stims, axis=0)
		if self.flatten:
			binned_stims = np.reshape(binned_stims, (T-n_frames, n_frames*width))
		else: 
			binned_stims = np.reshape(binned_stims, (T-n_frames, n_frames, width))
		spks_per_frm = spks_per_frm[n_frames:]
		if self.spk_thresh != None:
			spks_per_frm[spks_per_frm > self.spk_thresh] = float(self.spk_thresh)

		num_total = len(spks_per_frm)
		self.num_train = int(.7 * num_total)
		self.num_val = int(.1 * num_total)
		self.num_test = int(.2 * num_total)

		ind_array = np.asarray(range(self.num_train + self.num_val + self.num_test))
		
		if self.shuff:
			np.random.seed(324)
			np.random.shuffle(ind_array)
		    
		train_mask = ind_array[:self.num_train]
		val_mask = ind_array[self.num_train : self.num_train + self.num_val]
		test_mask = ind_array[self.num_train + self.num_val : self.num_train + self.num_val + self.num_test]

		self.X_train = binned_stims[train_mask]
		self.y_train = np.reshape(spks_per_frm[train_mask], [-1,1])
		self.X_val = binned_stims[val_mask]
		self.y_val = np.reshape(spks_per_frm[val_mask], [-1,1])
		self.X_test = binned_stims[test_mask]
		self.y_test = np.reshape(spks_per_frm[test_mask], [-1,1])

		self.data_dim = self.X_train.shape[1]

	def save_data(self):
		cc = self.cell[0] # either 's' or 'c'
		if cc == 'c':
			path = './Data/RustV1dat/'
		else:
			path = './Data/RGC_data/'
		if self.shuff:
			np.savetxt(path + 'X_train_' + cc + '1.csv', self.X_train, delimiter=',')
			np.savetxt(path + 'y_train_' + cc + '1.csv', self.y_train, delimiter=',')
			np.savetxt(path + 'X_val_' + cc + '1.csv', self.X_val, delimiter=',')
			np.savetxt(path + 'y_val_' + cc + '1.csv', self.y_val, delimiter=',')
			np.savetxt(path + 'X_test_' + cc + '1.csv', self.X_test, delimiter=',')
			np.savetxt(path + 'y_test_' + cc + '1.csv', self.y_test, delimiter=',')
		else:
			np.savetxt(path + 'X_train_ns_' + cc + '1.csv', self.X_train, delimiter=',')
			np.savetxt(path + 'y_train_ns_' + cc + '1.csv', self.y_train, delimiter=',')
			np.savetxt(path + 'X_val_ns_' + cc + '1.csv', self.X_val, delimiter=',')
			np.savetxt(path + 'y_val_ns_' + cc + '1.csv', self.y_val, delimiter=',')
			np.savetxt(path + 'X_test_ns_' + cc + '1.csv', self.X_test, delimiter=',')
			np.savetxt(path + 'y_test_ns_' + cc + '1.csv', self.y_test, delimiter=',')
            
	def convert_psth(self, preds):
		sflen = .01
		fml = np.arange(0.,self.nstim*sflen, 0.00001)
		fml2 = np.zeros((len(fml),))
		steps = self.Mraster.shape[1]
		for i in range(self.nstim):
			fml2[int(sflen/.00001)*i : int(sflen/.00001)*i + int(sflen/.00001)] = preds[i]/sflen
		fml3 = np.zeros((steps),)
		for i in range(steps):
			fml3[i] = fml2[100*i]
		return fml3



	def get_train_batch(self, batch_size, bnum):
		"""Return the next `batch_size` examples from this data set."""
		#i = int(np.random.rand() * (self.num_train - batch_size)) 
		start = batch_size * bnum
		end = start + batch_size
		if end > self.num_train:
			end = self.num_train - 1
		return self.X_train[start:end,:], self.y_train[start:end]

	def shuffle(self):
		"""Shuffle training data"""
		perm = np.arange(self.num_train)
		np.random.shuffle(perm)
		self.X_train = self.X_train[perm]
		self.y_train = self.y_train[perm]

