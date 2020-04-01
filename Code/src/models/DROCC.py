import torch
import json
import sys

from src.models.optim.DROCC_trainer import DROCC_trainer

class DROCC:
    """
    Object defining a DROCC model able to train, validate and test it but also
    to manage the output saving.
    """
    def __init__(self, net, r):
        """
        Built a DROCC instance with the given network and radius.
        ----------
        INPUT
            |---- net (nn.Module) the network to use in the DROCC model. it must
            |           output logit for the binary classification normal vs abnormal.
            |---- r (float) the radius defining the normal manifold around normal points.
        OUTPUT
            |---- None
        """
        self.net = net
        self.r = r
        self.trainer = None
        # to store the training point embdedding
        self.train_point_embed = None

        # Dict to store all the results : reconstruction and embedding
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

    def train(self, dataset, gamma=0.5, mu=0.5, lr=1e-4, lr_adv=1e-2, lr_milestone=(),
              weight_decay=1e-6, n_epoch=100, n_epoch_init=15, n_epoch_adv=15,
              batch_size=16, device='cuda', n_jobs_dataloader=0, LFOC=False, print_batch_progress=False):
        """
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to train the DeepSAD.
            |           Must return (input, label, mask, semi_label, idx).
            |---- gamma (float) the fraction of the radius defining the lower
            |           bound of the close adversarial samples layer.
            |---- mu (float) the weight given to adversarial samples in the loss.
            |---- lr (float) the learning rate.
            |---- lr_adv (float) the learning rate for the gradient ascent.
            |---- lr_milestone (tuple) the lr update steps.
            |---- weight_decay (float) the weight_decay for the Adam optimizer.
            |---- n_epoch (int) the total number of epoch.
            |---- n_epoch_init (int) the number of epoch without adversarial search.
            |---- n_epoch_adv (int) the number of epoch for the gradient ascent.
            |---- batch_size (int) the batch_size to use.
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- LFOC (bool) whether to use the DROCC-LF algorithm.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        self.trainer = DROCC_trainer(self.r, gamma=gamma, mu=mu, lr=lr, lr_adv=lr_adv,
                            lr_milestone=lr_milestone, weight_decay=weight_decay,
                            n_epoch=n_epoch, n_epoch_init=n_epoch_init, n_epoch_adv=n_epoch_adv,
                            batch_size=batch_size, device=device, n_jobs_dataloader=n_jobs_dataloader,
                            LFOC=LFOC, print_batch_progress=print_batch_progress)
        # train DROCC
        self.net = self.trainer.train(dataset, self.net)
        # get the results
        self.results['train']['time'] = self.trainer.train_time
        self.results['train']['loss'] = self.trainer.train_loss

    def validate(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Validate the DROCC model on the provided dataset with the provided parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to validate the DROCC.
            |           Must return (input, label, mask, semi_label, idx).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = DROCC_trainer(self.r, device=device,
                            n_jobs_dataloader=n_jobs_dataloader,
                            print_batch_progress=print_batch_progress)
        # validate DROCC
        self.trainer.validate(dataset, self.net)
        # get results
        self.results['valid']['time'] = self.trainer.valid_time
        self.results['valid']['auc'] = self.trainer.valid_auc
        self.results['valid']['f1'] = self.trainer.valid_f1
        self.results['valid']['scores'] = self.trainer.valid_scores
        self.results['scores_threshold'] = self.trainer.scores_threshold

    def test(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """
        Test the DROCC model on the provided dataset with the provided parameters.
        ----------
        INPUT
            |---- dataset (pytorch Dataset) the dataset on which to test the DROCC.
            |           Must return (input, label, mask, semi_label, idx).
            |---- device (str) the device to work on ('cpu' or 'cuda').
            |---- n_jobs_dataloader (int) number of workers for the dataloader.
            |---- print_batch_progress (bool) whether to display a progress bar.
        OUTPUT
            |---- None
        """
        if self.trainer is None:
            self.trainer = DROCC_trainer(self.r, device=device,
                            n_jobs_dataloader=n_jobs_dataloader,
                            print_batch_progress=print_batch_progress)
        # validate DROCC
        self.trainer.test(dataset, self.net)
        # get results
        self.results['test']['time'] = self.trainer.test_time
        self.results['test']['auc'] = self.trainer.test_auc
        self.results['test']['f1'] = self.trainer.test_f1
        self.results['test']['scores'] = self.trainer.test_scores

    def save_results(self, export_json_path):
        """
        Save the DROCC results (train time, test/validation time, test/validation AUC
        test/validation scores (loss and label for each samples)) as json.
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
        Save the DROCC model (state dict) on disk.
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
        Load the DROCC model (state dict) from the provided path.
        --------
        INPUT
            |---- model_path (str) filename of the model to load.
            |---- map_location (str) device on which to load.
        OUTPUT
            |---- None
        """
        model = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model['net_dict'])
