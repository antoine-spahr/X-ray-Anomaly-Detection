import torch
import json
import sys

from src.models.optim.autoencoder_trainer import AutoEncoderTrainer
from src.models.optim.DeepSAD_trainer import DeepSADTrainer

class DeepSAD:
    """
    Define a DeepSAD instance (inspired form the work of Lukas Ruff et al. (2019))
    and utility method to train, pretrain and test it.
    """
    def __init__(self, net, ae_net=None, eta=1.0):
        """
        Build the DeepSAD instance.
        ---------
        INPUT
            |---- net (nn.Module) the Encoder network to use.
            |---- ae_net (nn.Module) the Autoencoder network to use for pretraining.
            |---- eta (float) the DeepSAD parameter defining the weigth given to
            |           unspervized vs supervized sample in the loss. 1.0 gives
            |           equal weight to known and unknown samples. <1.0 gives more
            |           weight to unkonwn sample. >1.0 gives more weight to known
            |           samples.
        OUTPUT
            |---- None
        """
        self.eta = eta # balance of importance of labeled or unlabeled sample
        self.c = None # hypersphere center

        self.net = net
        self.trainer = None

        self.ae_net = ae_net
        self.ae_trainer = None

        # Dict to store all the results : reconstruction and embedding
        self.results = {
            'reconstruction':{
                'train':{
                    'time': None,
                    'loss': None
                },
                'scores_threshold':None,
                'valid':{
                    'time':None,
                    'auc':None,
                    'f1':None,
                    'scores':None
                },
                'test':{
                    'time':None,
                    'auc':None,
                    'f1':None,
                    'scores':None
                }
            },
            'embedding':{
                'train':{
                    'time': None,
                    'loss': None
                },
                'scores_threshold':None,
                'valid':{
                    'time':None,
                    'auc':None,
                    'f1':None,
                    'scores':None
                },
                'test':{
                    'time':None,
                    'auc':None,
                    'f1':None,
                    'scores':None
                }
            }
        }

    def train(self, dataset, lr=0.0001, n_epoch=150, lr_milestone=(), batch_size=64,
              weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Train the DeepSAD model on the provided dataset with the provided parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to train the DeepSAD.
            |           Must return (input, label, mask, semi_label, idx).
            |---- lr (float) the learning rate.
            |---- n_epoch (int) the number of epoch.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        self.trainer = DeepSADTrainer(self.c, self.eta, lr=lr, n_epoch=n_epoch, lr_milestone=lr_milestone,
                                      batch_size=batch_size, weight_decay=weight_decay, device=device,
                                      n_jobs_dataloader=n_jobs_dataloader, print_batch_progress=print_batch_progress)
        # train deepSAD
        self.net = self.trainer.train(dataset, self.net)
        # get results and parameters
        self.results['embedding']['train']['time'] = self.trainer.train_time
        self.results['embedding']['train']['loss'] = self.trainer.train_loss
        self.c = self.trainer.c.cpu().data.numpy().tolist()

    def validate(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Validate the DeepSAD model on the provided dataset with the provided
        parameters and find the best threshold ons scores that maximize the
        F1-score.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to validate the DeepSAD.
            |           Must return (input, label, mask, semi_label, idx).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device,
                                          n_jobs_dataloader=n_jobs_dataloader,
                                          print_batch_progress=print_batch_progress)

        self.trainer.validate(dataset, self.net)
        # recover restults
        self.results['embedding']['scores_threshold'] = self.trainer.scores_threhold
        self.results['embedding']['valid']['time'] = self.trainer.valid_time
        self.results['embedding']['valid']['auc'] = self.trainer.valid_auc
        self.results['embedding']['valid']['f1'] = self.trainer.valid_f1
        self.results['embedding']['valid']['scores'] = self.trainer.valid_scores

    def test(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Test the DeepSAD model on the provided dataset with the provided parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to test the DeepSAD.
            |           Must return (input, label, mask, semi_label, idx).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device,
                                          n_jobs_dataloader=n_jobs_dataloader,
                                          print_batch_progress=print_batch_progress)

        self.trainer.test(dataset, self.net)
        # recover restults
        self.results['embedding']['test']['time'] = self.trainer.test_time
        self.results['embedding']['test']['auc'] = self.trainer.test_auc
        self.results['embedding']['test']['f1'] = self.trainer.test_f1
        self.results['embedding']['test']['scores'] = self.trainer.test_scores

    def pretrain(self, train_dataset, valid_dataset, test_dataset, lr=0.0001, n_epoch=150, lr_milestone=(),
                 batch_size=64, weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Pretrain the DeepSAD model through the training of an Autoencoder on the
        provided dataset with the provided parameters.
        ----------
        INPUT
            |---- train_dataset (pytorch Dataset) the dataset on which to train
            |           the Autoencoder. Must return (input, label, mask,
            |           semi_label, idx).
            |---- test_dataset (pytorch Dataset) the dataset on which to test
            |           the Autoencoder. Must return (input, label, mask,
            |           semi_label, idx).
            |---- lr (float) the learning rate.
            |---- n_epoch (int) the number of epoch.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        self.ae_trainer = AutoEncoderTrainer(lr=lr, n_epoch=n_epoch, lr_milestone=lr_milestone,
                                      batch_size=batch_size, weight_decay=weight_decay, device=device,
                                      n_jobs_dataloader=n_jobs_dataloader, print_batch_progress=print_batch_progress)
        # Train AE
        self.ae_net = self.ae_trainer.train(train_dataset, self.ae_net)
        self.results['reconstruction']['train']['time'] = self.ae_trainer.train_time
        self.results['reconstruction']['train']['loss'] = self.ae_trainer.train_loss

        # Validate AE
        self.ae_trainer.validate(valid_dataset, self.ae_net)
        #self.scores_threhold_rec = self.ae_trainer.scores_threhold # get best threshold for max F1-score
        self.results['reconstruction']['scores_threshold'] = self.ae_trainer.scores_threhold
        self.results['reconstruction']['valid']['time'] = self.ae_trainer.valid_time
        self.results['reconstruction']['valid']['auc'] = self.ae_trainer.valid_auc
        self.results['reconstruction']['valid']['f1'] = self.ae_trainer.valid_f1
        self.results['reconstruction']['valid']['scores'] = self.ae_trainer.valid_scores

        # Test AE
        self.ae_trainer.test(test_dataset, self.ae_net)
        self.results['reconstruction']['test']['time'] = self.ae_trainer.test_time
        self.results['reconstruction']['test']['auc'] = self.ae_trainer.test_auc
        self.results['reconstruction']['test']['f1'] = self.ae_trainer.test_f1
        self.results['reconstruction']['test']['scores'] = self.ae_trainer.test_scores

        # Initialize DeepSAD model with Encoder's weights
        self.init_network_weights_from_pretrain()

    def init_network_weights_from_pretrain(self):
        """
        Initialize DeepSAD encoder weights with the autoencoder's one.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()
        # filter elements of the AE out to keep only those matching DeepSAD's one
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Update the DeepSAD state dict
        net_dict.update(ae_net_dict)
        self.net.load_state_dict(net_dict)

    def save_model(self, export_path, save_ae=True):
        """
        Save the model (hypersphere center, DeepSAD state dict, (Autoencoder
        state dict)) on disk.
        ----------
        INPUT
            |---- export_path (str) the filename where to export the model.
            |---- save_ae (bool) whether to save the Autoencoder state dict.
        OUTPUT
            |---- None
        """
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None
        #ae_threshold = self.scores_threhold_rec if save_ae else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_path)

    def load_model(self, model_path, load_ae=True, map_location='cpu'):
        """
        Load the model (hypersphere center, DeepSAD state dict, (Autoencoder
        state dict) from the provided path.
        --------
        INPUT
            |---- model_path (str) filename of the model to load.
            |---- load_ae (bool) whether to load the Autoencoder state dict.
            |---- map_location (str) device on which to load.
        OUTPUT
            |---- None
        """
        assert load_ae and (self.ae_net is not None), 'The trainer has not been initialized with an Autoencoder. It can thus not be loaded.'
        model_dict = torch.load(model_path, map_location=map_location)
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        if load_ae and (self.ae_net is not None):
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json_path):
        """
        Save the DeepSAD results (train time, test time, test AUC, test scores
        (loss and label for each samples)) as json.
        ----------
        INPUT
            |---- export_json_path (str) the json filename where to save.
        OUTPUT
            |---- None
        """
        with open(export_json_path, 'w') as f:
            json.dump(self.results, f)
