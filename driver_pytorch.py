import gzip
import pickle
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from RVAE import RVAE

import numpy as np
from numpy.linalg import cholesky

# def train(loader, model, params ):
def train(train_loader, model, optimizer, epoch, batch_size, clip, print_every, normalize=False):
	train_loss = 0
	for batch_idx, data in enumerate(train_loader):

		data = Variable(data.squeeze().transpose(0,1))

		if normalize:
			data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())		

		optimizer.zero_grad()

		kld_loss, nll_loss, _, _ = model(data)

		loss = kld_loss + nll_loss
		loss.backward()
		optimizer.step()

		nn.utils.clip_grad_norm(model.parameters(), clip)

		if batch_idx % print_every == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
			    epoch, batch_idx * len(data), len(train_loader.dataset),
			    100. * batch_idx / len(train_loader),
			    kld_loss.data.item() / batch_size,
			    nll_loss.data.item() / batch_size))

			sample = model.sample(28)
			plt.imshow(sample.numpy())
			plt.pause(1e-6)

		train_loss += loss.data.item()

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))

def test(test_loader, model, epoch, normalize=False):
	"""uses test data to evaluate 
	likelihood of the model"""

	mean_kld_loss, mean_nll_loss = 0, 0
	for i, data in enumerate(test_loader):                                            

		data = Variable(data.squeeze().transpose(0, 1))

		if normalize:
			data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())

		kld_loss, nll_loss, _, _ = model(data)
		mean_kld_loss += kld_loss.data.item()
		mean_nll_loss += nll_loss.data.item()

	mean_kld_loss /= len(test_loader.dataset)
	mean_nll_loss /= len(test_loader.dataset)

	print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
		mean_kld_loss, mean_nll_loss))

def simulated_data_loaders(num_features, num_trials, T, train_ratio=.8, seed=47):
	np_rng = np.random.RandomState(seed)
	raw_samp = np_rng.uniform(0.025, 0.035, (num_trials, 1))
	z = np.hstack(raw_samp + np_rng.normal(0., .0025, (num_trials,1)) for _ in range(num_features))
	z[z <= .001] = .001
	z = z.reshape(int(num_trials/T), T, num_features).astype('float32')
	train_ind = int(train_ratio * z.shape[0])
	z_train = z[:train_ind, :, :]
	z_test = z[train_ind:, :, :]
	train_loader = torch.utils.data.DataLoader( z_train, batch_size=T, shuffle=True)
	test_loader = torch.utils.data.DataLoader( z_test, batch_size=T, shuffle=True)
	return train_loader, test_loader

def mnist_loaders(batch_size):
	data_path = '/home/charles/src/Python/data/mnist.pkl.gz'
	with gzip.open(data_path, 'rb') as f:
		train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
	train_data = train_set[0].reshape(50000, 28, 28)
	test_data = test_set[0].reshape(10000, 28, 28)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
	return train_loader, test_loader

def main(mnist=True, normalize=False, save_model=False):

	if mnist:
		# Params
		num_features        = 28
		batch_size          = 100
		n_phi_x_hidden      = 150
		n_phi_z_hidden      = 150
		n_latent            = 28
		n_hidden_prior      = 150
		n_rec_hidden        = 150
		n_encoder_hidden    = 150
		n_decoder_hidden    = 150
		n_rec_layers        = 3
		bias                = False
		
		# Loaders
		train_loader, test_loader = mnist_loaders(batch_size)
		
	else:
		# Params
		num_features        = 24
		batch_size          = 500
		num_trials          = 500*100
		T                   = 500
		n_phi_x_hidden      = 20
		n_phi_z_hidden      = 20
		n_latent            = 18
		n_hidden_prior      = 20
		n_rec_hidden        = 20
		n_encoder_hidden    = 20
		n_decoder_hidden    = 20
		n_rec_layers        = 50
		bias                = False
		train_ratio         = .8

		# Loaders
		train_loader, test_loader = simulated_data_loaders(num_features,
							   num_trials,
							   T,
							   train_ratio)							   

	# Hyperparameters
	seed = 47
	n_epochs = 100
	clip = 1
	learning_rate = 1e-3
	gseed = 128
	print_every = 100
	save_every = 10
	
	torch.manual_seed(seed)
	plt.ion()
	
	model = RVAE(num_features,
		     n_phi_x_hidden,
		     n_phi_z_hidden,
		     n_latent,
		     n_hidden_prior,
		     n_rec_hidden,
		     n_encoder_hidden,
		     n_decoder_hidden,
		     n_rec_layers,
		     bias
		     )

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	for epoch in range(1, n_epochs + 1):

		train(train_loader, model, optimizer, epoch, batch_size, clip, print_every, normalize=False)
		test(test_loader, model, epoch, normalize=False)
		
		if save_model and epoch % save_every == 1:
			fn = 'models/vrnn_state_dict_'+str(epoch)+'.pth'
			torch.save(model.state_dict(), fn)
			print('Saved model to '+fn)
			
if __name__ == '__main__':
	main(mnist=True, normalize=False)
