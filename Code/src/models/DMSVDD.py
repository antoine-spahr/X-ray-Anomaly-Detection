import torch
import json
import sys

from src.models.optim.DMSVDD_Joint_trainer import DMSVDD_Joint_trainer

class joint_DMSVDD:
    """

    """
    def __init__(self, net, nu=0.05, soft_boundary=False):
        """

        """
        self.R = None # list of sphere radius
        self.soft_boundary = soft_boundary
        self.nu = nu # balance of importance of labeled or unlabeled sample
        self.c = None # hyperspheres centers (N_spheres x Embed_dim)

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

    def train(self, dataset, lr=1e-4, n_epoch=150, n_epoch_pretrain=10, n_epoch_warm_up=5,
              n_sphere_init=100, lr_milestone=(), batch_size=64, weight_decay=1e-6,
              device='cuda', n_jobs_dataloader=0, print_batch_progress=False,
              criterion_weight=(0.5,0.5)):
        """
        Train the joint DMSVDD model on the provided dataset with the provided
        parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to train the DeepSVDD.
            |           Must return (input, label, mask, semi_label, idx).
            |---- lr (float) the learning rate.
            |---- n_epoch (int) the number of epoch.
            |---- n_epoch_pretrain (int) the number of epoch to perform with only
            |           the reconstruction loss.
            |---- n_epoch_warm_up (int) number of epoch without soft boundaries.
            |---- n_sphere_init (int) the number of initial hypersphere.
            |---- lr_milestone (tuple) the lr update steps.
            |---- batch_size (int) the batch_size to use.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
            |---- criterion_weight (tuple (weight reconstruction, weight MSVDD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           MSVDD).
        OUTPUT
            |---- None
        """
        self.trainer = DMSVDD_Joint_trainer(self.c, self.nu, self.R, lr=lr,
                                n_epoch=n_epoch, n_epoch_pretrain=n_epoch_pretrain,
                                n_epoch_warm_up=n_epoch_warm_up, n_sphere_init=n_sphere_init,
                                lr_milestone=lr_milestone, batch_size=batch_size,
                                weight_decay=weight_decay, device=device,
                                n_jobs_dataloader=n_jobs_dataloader,
                                print_batch_progress=print_batch_progress,
                                criterion_weight=criterion_weight,
                                soft_boundary=self.soft_boundary)

        # pretrain AE
        if n_epoch_pretrain > 0:
            self.net = self.trainer.pretrain(dataset, self.net)
            self.results['pretrain']['time'] = self.trainer.pretrain_time
            self.results['pretrain']['loss'] = self.trainer.pretrain_loss

        # train Joint DeepSVDD
        self.net = self.trainer.train(dataset, self.net)
        # get results
        self.results['train']['time'] = self.trainer.train_time
        self.results['train']['loss'] = self.trainer.train_loss
        self.c = self.trainer.c.cpu().data.numpy()
        self.R = self.trainer.R.cpu().data.numpy()

    def validate(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False,
                 criterion_weight=(0.5, 0.5)):
        """
        Validate the Joint DMSVDD model on the provided dataset with the provided
        parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to valid the DMSVDD.
            |           Must return (input, label, mask, semi_label, idx).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
            |---- criterion_weight (tuple (weight reconstruction, weight MSVDD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           MSVDD).
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = DMSVDD_Joint_trainer(self.c, self.nu, self.R, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader,
                                    print_batch_progress=print_batch_progress,
                                    criterion_weight=criterion_weight,
                                    soft_boundary = self.soft_boundary)
        # test the network
        self.trainer.validate(dataset, self.net)
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


    def test(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False,
             criterion_weight=(0.5, 0.5)):
        """
        Test the Joint DMSVDD model on the provided dataset with the provided
        parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to test the DMSVDD.
            |           Must return (input, label, mask, semi_label, idx).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
            |---- criterion_weight (tuple (weight reconstruction, weight MSVDD))
            |           the weighting of the two losses (masked MSE loss of the
            |           AutoEncoder and the hypersphere center distance for the
            |           MSVDD).
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = DMSVDD_Joint_trainer(self.c, self.nu, self.R, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader,
                                    print_batch_progress=print_batch_progress,
                                    criterion_weight=criterion_weight,
                                    soft_boundary = self.soft_boundary)
        # test the network
        self.trainer.test(dataset, self.net)
        # get results
        self.results['embedding']['test']['time'] = self.trainer.test_time
        self.results['embedding']['test']['auc'] = self.trainer.test_auc_ad
        self.results['embedding']['test']['f1'] = self.trainer.test_f1_ad
        self.results['embedding']['test']['scores'] = self.trainer.test_scores_ad
        self.results['reconstruction']['test']['time'] = self.trainer.test_time
        self.results['reconstruction']['test']['auc'] = self.trainer.test_auc_rec
        self.results['reconstruction']['test']['f1'] = self.trainer.test_f1_rec
        self.results['reconstruction']['test']['scores'] = self.trainer.test_scores_rec

    def save_results(self, export_json_path):
        """
        Save the Joint DMSVDD results (train time, test time, test AUC
        (reconstruction and DMSVDD), test scores (loss and label for each
        samples)) as json.
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
        Save the model (hypersphere centers, radius, Joint DMSVDD state dict) on disk.
        ----------
        INPUT
            |---- export_path (str) the filename where to export the model.
        OUTPUT
            |---- None
        """
        net_dict = self.net.state_dict()
        torch.save({'c':self.c,
                    'R':self.R,
                    'net_dict':net_dict}, export_path)

    def load_model(self, model_path, map_location='cpu'):
        """
        Load the model (hypersphere center, Joint DMSVDD state dict) from the
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
        self.R = model['R']
        self.c = model['c']
