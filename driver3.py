import gzip
import pickle
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from RVAE3 import RVAE

import numpy as np
from numpy.linalg import cholesky

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

def train(epoch):
	train_loss = 0
	for batch_idx, data in enumerate(train_loader):

            data = Variable(data.squeeze().transpose(0,1))
            ##############################################
            ### At this point data has shape (28, 128, 28)
            ##############################################

            # data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())		
            
            #forward + backward + optimize
            optimizer.zero_grad()
		
            kld_loss, nll_loss, _, _ = model(data)
            loss = kld_loss + nll_loss
            loss.backward()
            optimizer.step()
            
            #grad norm clipping, only in pytorch version >= 1.10
            nn.utils.clip_grad_norm(model.parameters(), clip)
            
            #printing
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


def test(epoch):
	"""uses test data to evaluate 
	likelihood of the model"""
	
	mean_kld_loss, mean_nll_loss = 0, 0
	for i, data in enumerate(test_loader):                                            
		
		#data = Variable(data)
		data = Variable(data.squeeze().transpose(0, 1))
		data = (data - data.min().data.item()) / (data.max().data.item() - data.min().data.item())

		kld_loss, nll_loss, _, _ = model(data)
		mean_kld_loss += kld_loss.data.item()
		mean_nll_loss += nll_loss.data.item()

	mean_kld_loss /= len(test_loader.dataset)
	mean_nll_loss /= len(test_loader.dataset)

	print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
		mean_kld_loss, mean_nll_loss))


num_features = 4
batch_size = 100
num_trials = 4000000
T = 128
n_phi_x_hidden = 4
n_phi_z_hidden = 4
n_latent = 4
n_hidden_prior = 4
n_rec_hidden = 4
n_encoder_hidden = 4
n_decoder_hidden = 4
n_rec_layers = 1
bias = False 
#######
seed = 47
n_epochs = 3
clip = 100
learning_rate = 1e-3
gseed = 128
print_every = 100
save_every = 10

torch.manual_seed(seed)
plt.ion()

np_rng = np.random.RandomState(44)
raw_samp = np_rng.uniform(0.025, 0.035, (num_trials, 1))
z = np.hstack(raw_samp + np_rng.normal(0., .0025, (num_trials,1)) for _ in range(num_features))
z[z <= .001] = .001
z = z.reshape(int(num_trials/T), T, num_features).astype('float32')
train_ind = int(.8 * z.shape[0])
z_train = z[:train_ind, :, :]
z_test = z[train_ind:, :, :]
train_loader = torch.utils.data.DataLoader( z_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader( z_test, batch_size=batch_size, shuffle=True)

# data_path = '/home/charles/src/Python/data/mnist.pkl.gz'
# with gzip.open(data_path, 'rb') as f:
#     train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
# train_data = train_set[0].reshape(50000, 28, 28)
# test_data = test_set[0].reshape(10000, 28, 28)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

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
	
	#training + testing
	train(epoch)
	test(epoch)

	#saving model
	if epoch % save_every == 1:
		fn = 'models/vrnn_state_dict_'+str(epoch)+'.pth'
		torch.save(model.state_dict(), fn)
		print('Saved model to '+fn)
