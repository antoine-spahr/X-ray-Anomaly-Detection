import torch
import json
import sys

from src.models.optim.DMSAD_Joint_trainer import DMSAD_Joint_trainer

class joint_DMSAD:
    """
    Define a semi-supervised Deep Multi-sphere SAD model based on the
    implementation of Ruff et al. 2020 and Ghafoori et al. 2020.
    """
    def __init__(self, net, eta=1.0, gamma=0.05):
        """
        Build a DMSAD for the provided network architecture.
        ----------
        INPUT
            |---- net (nn.Module) the network architecture for the model. It
            |           should return two embdedings : one for the AE reconstruction
            |           and one for the hypershere representation.
            |---- eta (float) positive number defining the importance of the
            |           semi-supervised samples in the loss contribution.
            |---- gamma (float) the fraction of allowed outlier when setting the
            |           radius of each sphere in the end.
        OUTPUT
            |---- None
        """
        self.c = None
        self.R = None
        self.eta = eta
        self.gamma = gamma
        self.net = net
        self.trainer = None

        # Dict to store all the results : reconstruction and embedding
        self.results = {
            'pretrain':{
                'time': None,
                'loss': None
            },
            'train':{
                'time': None,
                'loss': None
            },
            'reconstruction':{
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

    def train(self, dataset, valid_dataset=None, n_sphere_init=100, n_epoch=150,
              n_epoch_pretrain=10, lr=1e-4, weight_decay=1e-6, lr_milestone=(),
              criterion_weight=(0.5, 0.5), reset_scaling_epoch=3, batch_size=64,
              n_jobs_dataloader=0, device='cuda', print_batch_progress=False):
        """
        Train the DMSAD model with the passed settings.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to train the DMSAD.
            |           Must return (input, label, mask, semi_label, idx).)
            |---- valid_dataset (pytorch Dataset) the dataset on which to validate
            |           the model at each epoch. No validation is computed is not
            |           provided. Must return (input, label, mask, semi_label, idx).)
            |---- n_sphere_init (int) the number of initial hypersphere.
            |---- n_epoch (int) the number of epoch.
            |---- n_epoch_pretrain (int) the number of epoch to perform with only
            |           the reconstruction loss.
            |---- lr (float) the learning rate.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- lr_milestone (tuple) the lr update steps (90% recudtion at each step).
            |---- criterion_weight (tuple (weight reconstruction, weight MSVDD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           MSAD).
            |---- reset_scaling_epoch (int) the epoch at which the scling factor
            |           should be recomputed. To avoid get a more meaningful
            |           scalling after few spheres were removed.
            |---- batch_size (int) the batch_size to use.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        self.trainer = DMSAD_Joint_trainer(self.c, self.R, eta=self.eta, gamma=self.gamma,
                            n_sphere_init=n_sphere_init, n_epoch=n_epoch,
                            n_epoch_pretrain=n_epoch_pretrain, lr=lr,
                            weight_decay=weight_decay, lr_milestone=lr_milestone,
                            criterion_weight=criterion_weight,
                            reset_scaling_epoch=reset_scaling_epoch, batch_size=batch_size,
                            n_jobs_dataloader=n_jobs_dataloader, device=device,
                            print_batch_progress=print_batch_progress)

        # pretrain with only reconstruction loss
        if n_epoch_pretrain > 0:
            self.net = self.trainer.pretrain(self.net, dataset)
            self.results['pretrain']['time'] = self.trainer.pretrain_time
            self.results['pretrain']['loss'] = self.trainer.pretrain_loss

        # train
        self.net = self.trainer.train(self.net, dataset, valid_dataset=valid_dataset)
        # get results and centers
        self.results['train']['time'] = self.trainer.train_time
        self.results['train']['loss'] = self.trainer.train_loss
        self.c = self.trainer.c.cpu().data.numpy()
        self.R = self.trainer.R.cpu().data.numpy()

    def validate(self, dataset, batch_size=64, n_jobs_dataloader=0,
                criterion_weight=(0.5, 0.5), device='cuda', print_batch_progress=False):
        """
        Validate the DMSAD model with the passed settings.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to validate DMSAD.
            |           Must return (input, label, mask, semi_label, idx).)
            |---- batch_size (int) the batch_size to use.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- criterion_weight (tuple (weight reconstruction, weight MSVDD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           MSAD).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = DMSAD_Joint_trainer(self.c, self.R, self.eta, self.gamma,
                                batch_size=batch_size, n_jobs_dataloader=n_jobs_dataloader,
                                criterion_weight=criterion_weight, device=device,
                                print_batch_progress=print_batch_progress)

        self.trainer.evaluate(self.net, dataset, mode='valid', final=True)
        # get results
        self.results['embedding']['valid']['time'] = self.trainer.valid_time
        self.results['embedding']['valid']['auc'] = self.trainer.valid_auc_ad
        self.results['embedding']['valid']['f1'] = self.trainer.valid_f1_ad
        self.results['embedding']['valid']['scores'] = self.trainer.valid_scores_ad
        self.results['embedding']['scores_threshold'] = self.trainer.scores_threhold_ad
        self.results['reconstruction']['valid']['time'] = self.trainer.valid_time
        self.results['reconstruction']['valid']['auc'] = self.trainer.valid_auc_rec
        self.results['reconstruction']['valid']['f1'] = self.trainer.valid_f1_rec
        self.results['reconstruction']['valid']['scores'] = self.trainer.valid_scores_rec
        self.results['reconstruction']['scores_threshold'] = self.trainer.scores_threhold_rec

    def test(self, dataset, batch_size=64, n_jobs_dataloader=0,
             criterion_weight=(0.5, 0.5), device='cuda', print_batch_progress=False):
        """
        Test the DMSAD model with the passed settings.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to test DMSAD.
            |           Must return (input, label, mask, semi_label, idx).)
            |---- batch_size (int) the batch_size to use.
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- criterion_weight (tuple (weight reconstruction, weight MSVDD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           MSAD).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = DMSAD_Joint_trainer(self.c, self.R, self.eta, self.gamma,
                                batch_size=batch_size, n_jobs_dataloader=n_jobs_dataloader,
                                criterion_weight=criterion_weight, device=device,
                                print_batch_progress=print_batch_progress)

        self.trainer.evaluate(self.net, dataset, mode='test')
        # get results
        self.results['embedding']['test']['time'] = self.trainer.test_time
        self.results['embedding']['test']['auc'] = self.trainer.test_auc_ad
        self.results['embedding']['test']['f1'] = self.trainer.test_f1_ad
        self.results['embedding']['test']['scores'] = self.trainer.test_scores_ad
        self.results['reconstruction']['test']['time'] = self.trainer.test_time
        self.results['reconstruction']['test']['auc'] = self.trainer.test_auc_rec
        self.results['reconstruction']['test']['f1'] = self.trainer.test_f1_rec
        self.results['reconstruction']['test']['scores'] = self.trainer.test_scores_rec

    def save_model(self, export_path):
        """
        Save the Joint DMSAD results (train time, test time, test AUC
        (reconstruction and DMSAD), test scores (loss and label for each
        samples)) as json.
        ----------
        INPUT
            |---- export_json_path (str) the json filename where to save.
        OUTPUT
            |---- None
        """
        torch.save({'c': self.c,
                    'R': self.R,
                    'net_dict': self.net.state_dict()}, export_path)

    def save_results(self, export_json_path):
        """
        Save the model (hypersphere centers and Joint DMSAD state dict) on disk.
        ----------
        INPUT
            |---- export_path (str) the filename where to export the model.
        OUTPUT
            |---- None
        """
        with open(export_json_path, 'w') as f:
            json.dump(self.results, f)

    def load_model(self, model_path, map_location='cpu'):
        """
        Load the model (hypersphere centers, Joint DMSAD state dict) from the
        provided path.
        --------
        INPUT
            |---- model_path (str) filename of the model to load.
            |---- map_location (str) device on which to load.
        OUTPUT
            |---- None
        """
        model = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model['net_dict'])
        self.c = model['c']
        self.R = model['R']
