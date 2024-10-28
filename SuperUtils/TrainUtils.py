import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


class Trainer:
	def __init__(self, config, net, trainloader=None, testloader=None, save=False, input_helper_function=None, target_helper_function=None, score_helper_function=None, save_on_folder=None):
		self.net = net
		self.config = config 
		self.trainloader = trainloader 
		self.testloader = testloader
		self.save = save
		self.input_helper_function = input_helper_function
		self.target_helper_function = target_helper_function
		self.score_helper_function = score_helper_function
		self.save_on_folder = save_on_folder

	def train(self):
		# TRAIN
		loss_fn = eval(f"nn.{self.config['loss_fn']}()")
		opt = optim.Adam(self.net.parameters(), lr=self.config['lr'])
		for epoch in tqdm( range(1, self.config['epochs']+1) ):
			epoch_loss = []
			if self.testloader: epoch_val_loss = []
			if self.score_helper_function: epoch_score = []
			self.net.train()
			for inps, lbls in self.trainloader:
				opt.zero_grad()
				if self.input_helper_function: inps = self.input_helper_function(inps)
				if self.target_helper_function: lbls = self.target_helper_function(lbls)
				inps = inps.to(self.config['device'])
				lbls = lbls.to(self.config['device'])
				outs = self.net(inps)
				if self.score_helper_function: epoch_score.append(self.score_helper_function(outs, lbls))
				loss = loss_fn(outs, lbls)
				epoch_loss.append(loss.item())
				loss.backward()
				opt.step()
			if self.testloader:
				self.net.eval()
				with torch.inference_mode():
					for inps, lbls in self.testloader:
						if self.input_helper_function: inps = self.input_helper_function(inps)
						if self.target_helper_function: lbls = self.target_helper_function(lbls)
						inps = inps.to(self.config['device'])
						lbls = lbls.to(self.config['device'])
						outs = self.net(inps)
						loss = loss_fn(outs, lbls)
						epoch_val_loss.append(loss.item())
			if self.testloader:
				print(f'epoch {epoch} loss {sum(epoch_loss)/len(epoch_loss)} val_loss {sum(epoch_val_loss)/len(epoch_val_loss)}')
			else:
				print(f'epoch {epoch} loss {sum(epoch_loss)/len(epoch_loss)}')
			if self.score_helper_function: print(f'Score: {sum(epoch_score)/len(epoch_score)}')
		if self.save:
			if self.save_on_folder: torch.save(self.net.state_dict(), f'{save_on_folder}\\weights.pt')
			else: torch.save(self.net.state_dict(), 'weights.pt')


	def cross_validation_train(self, dataset, n_splits=5):
		kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
		fold_results = []

		# Loss function initialization
		loss_fn = eval(f"nn.{self.config['loss_fn']}()")

		# Iterate over each fold
		for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
			print(f'Fold {fold + 1}/{n_splits}')
            
			# Split dataset into training and validation subsets for the current fold
			train_subset = Subset(dataset, train_idx)
			val_subset = Subset(dataset, val_idx)

			# Create DataLoaders
			train_loader = DataLoader(train_subset, batch_size=self.config['batch_size'], shuffle=True)
			val_loader = DataLoader(val_subset, batch_size=self.config['batch_size'], shuffle=False)

			# Initialize the model and optimizer for this fold
			self.net.apply(self._reset_weights)  # Reset model weights
			opt = optim.Adam(self.net.parameters(), lr=self.config['lr'])

			# Training loop for each epoch in the current fold
			for epoch in tqdm(range(1, self.config['epochs'] + 1)):
				epoch_loss = []
				epoch_val_loss = []
				if self.score_helper_function: epoch_score = []

				# Training phase
				self.net.train()
				for inps, lbls in train_loader:
					opt.zero_grad()
					if self.input_helper_function: inps = self.input_helper_function(inps)
					if self.target_helper_function: lbls = self.target_helper_function(lbls)
					inps = inps.to(self.config['device'])
					lbls = lbls.to(self.config['device'])
					outs = self.net(inps)
					if self.score_helper_function: epoch_score.append(self.score_helper_function(outs, lbls))
					loss = loss_fn(outs, lbls)
					epoch_loss.append(loss.item())
					loss.backward()
					opt.step()

				self.net.eval()
				with torch.inference_mode():
					for inps, lbls in val_loader:
						if self.input_helper_function: inps = self.input_helper_function(inps)
						if self.target_helper_function: lbls = self.target_helper_function(lbls)
						inps = inps.to(self.config['device'])
						lbls = lbls.to(self.config['device'])
						outs = self.net(inps)
						loss = loss_fn(outs, lbls)
						epoch_val_loss.append(loss.item())

				# Logging losses and scores
				print(f'epoch {epoch} loss {sum(epoch_loss)/len(epoch_loss)} val_loss {sum(epoch_val_loss)/len(epoch_val_loss)}')
				if self.score_helper_function:
					print(f'Score: {sum(epoch_score)/len(epoch_score)}')

			# Store average validation loss/score for the current fold
			avg_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
			fold_results.append(avg_val_loss)

		# Output cross-validation results
		avg_loss = sum(fold_results) / len(fold_results)
		print(f'Cross-validation average validation loss: {avg_loss}')
		return fold_results

	def _reset_weights(self, m):
		# Helper function to reset weights of the model before each fold
		if isinstance(m, (nn.Conv2d, nn.Linear)):
			m.reset_parameters()








