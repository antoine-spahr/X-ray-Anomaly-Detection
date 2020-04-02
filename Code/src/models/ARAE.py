import torch
import json
import sys

from src.models.optim.ARAE_trainer import ARAE_trainer

class ARAE:
    """
    Object defining a ARAE model able to train, validate and test it but also
    to manage the output saving.
    """
    def __init__(self, net, gamma, epsilon):
        """
        Built a DROCC instance with the given network and adversarial search
        settings gamma and epsilon.
        ----------
        INPUT
            |---- net (nn.Module) the network to use in the ARAE model. It must
            |           be an autoencoder able to output the latent emdeding and
            |           the reconstruction.
            |---- gamma (float) the weight of the adversarial loss.
            |---- epsilon (float) define l-inf bounds of the allowed adversarial
            |           perturbation of the normal inputs.
        OUTPUT
            |---- None
        """
        self.net = net
        self.gamma = gamma
        self.epsilon = epsilon

        # Dict to store all the results
        self.results = {
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

    def train(self, dataset, lr=1e-4, lr_adv=1e-2, lr_milestone=(), weight_decay=1e-6,
              n_epoch=100, n_epoch_adv=15, batch_size=16, device='cuda',
              n_jobs_dataloader=0, print_batch_progress=False):
        """
        Train the ARAE model.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to train the DeepSAD.
            |           Must return (input, label, mask, semi_label, idx).
            |---- lr (float) the learning rate.
            |---- lr_adv (float) the learning rate for the gradient ascent.
            |---- lr_milestone (tuple) the lr update steps.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- n_epoch (int) the total number of epoch.
            |---- n_epoch_adv (int) the number of epoch for the gradient ascent.
            |---- batch_size (int) the batch_size to use.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        self.trainer = ARAE_trainer(self.gamma, self.epsilon, lr=lr, lr_adv=lr_adv,
                                lr_milestone=lr_milestone, weight_decay=weight_decay,
                                n_epoch=n_epoch, n_epoch_adv=n_epoch_adv, batch_size=batch_size,
                                device=device, n_jobs_dataloader=n_jobs_dataloader,
                                print_batch_progress=print_batch_progress)
        # train ARAE
        self.net = self.trainer.train(dataset, self.net)
        # get results
        self.results['train']['time'] = self.trainer.train_time
        self.results['train']['loss'] = self.trainer.train_loss

    def validate(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Validate the ARAE model on the provided dataset with the provided parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to validate the ARAE.
            |           Must return (input, label, mask, semi_label, idx).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = ARAE_trainer(self.gamma, self.epsilon, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader,
                                    print_batch_progress=print_batch_progress)
        # validate ARAE
        self.trainer.validate(dataset, self.net)
        # get results
        self.results['valid']['time'] = self.trainer.valid_time
        self.results['valid']['auc'] = self.trainer.valid_auc
        self.results['valid']['f1'] = self.trainer.valid_f1
        self.results['valid']['scores'] = self.trainer.valid_scores
        self.results['scores_threshold'] = self.trainer.scores_threshold

    def test(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Test the ARAE model on the provided dataset with the provided parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to test the ARAE.
            |           Must return (input, label, mask, semi_label, idx).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = ARAE_trainer(self.gamma, self.epsilon, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader,
                                    print_batch_progress=print_batch_progress)
        # validate ARAE
        self.trainer.test(dataset, self.net)
        # get results
        self.results['test']['time'] = self.trainer.test_time
        self.results['test']['auc'] = self.trainer.test_auc
        self.results['test']['f1'] = self.trainer.test_f1
        self.results['test']['scores'] = self.trainer.test_scores

    def save_results(self, export_json_path):
        """
        Save the ARAE results (train time, test/validation time, test/validation
        AUC test/validation scores (loss and label for each samples)) as json.
        ----------
        INPUT
            |---- export_json_path (str) the json filename where to save.
        OUTPUT
            |---- None
        """
        with open(export_json_path, 'w') as f:
            json.dump(self.results, f)

    def save_model(self, export_path):
        """
        Save the ARAE model (state dict) on disk.
        ----------
        INPUT
            |---- export_path (str) the filename where to export the model.
        OUTPUT
            |---- None
        """
        net_dict = self.net.state_dict()
        torch.save({'net_dict':net_dict}, export_path)

    def load_model(self, model_path, map_location='cpu'):
        """
        Load the ARAE model (state dict) from the provided path.
        --------
        INPUT
            |---- model_path (str) filename of the model to load.
            |---- map_location (str) device on which to load.
        OUTPUT
            |---- None
        """
        model = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model['net_dict'])
