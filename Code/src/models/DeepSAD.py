import torch
import json
import sys

from src.models.optim.autoencoder_trainer import AutoEncoderTrainer
from src.models.optim.DeepSAD_trainer import DeepSADTrainer

class DeepSAD:
    """

    """
    def __init__(self, net, ae_net=None, eta=1.0):
        """

        """
        self.eta = eta # balance of importance of labeled or unlabeled sample
        self.c = None # hypersphere center

        self.net = net
        self.trainer = None

        self.ae_net = ae_net
        self.ae_trainer = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def train(self, dataset, lr=0.0001, n_epoch=150, lr_milestone=(), batch_size=64,
              weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """

        """
        self.trainer = DeepSADTrainer(self.c, self.eta, lr=lr, n_epoch=n_epoch, lr_milestone=lr_milestone,
                                      batch_size=batch_size, weight_decay=weight_decay, device=device,
                                      n_jobs_dataloader=n_jobs_dataloader, print_batch_progress=print_batch_progress)
        # train deepSAD
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()

    def test(self, dataset, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """

        """
        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device,
                                          n_jobs_dataloader=n_jobs_dataloader,
                                          print_batch_progress=print_batch_progress)

        self.trainer.test(dataset, self.net)
        # recover restults
        self.results['test_time'] = self.trainer.test_time
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, train_dataset, test_dataset, lr=0.0001, n_epoch=150, lr_milestone=(),
                 batch_size=64, weight_decay=1e-6, device='cuda', n_jobs_dataloader=0, print_batch_progress=False):
        """

        """
        self.ae_trainer = AutoEncoderTrainer(lr=lr, n_epoch=n_epoch, lr_milestone=lr_milestone,
                                      batch_size=batch_size, weight_decay=weight_decay, device=device,
                                      n_jobs_dataloader=n_jobs_dataloader, print_batch_progress=print_batch_progress)
        # Train AE
        self.ae_net = self.ae_trainer.train(train_dataset, self.ae_net)
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test AE
        self.ae_trainer.test(test_dataset, self.ae_net)
        self.ae_results['test_time'] = self.ae_trainer.test_time
        self.ae_results['test_auc'] = self.ae_trainer.test_auc

        # Initialize DeepSAD model with Encoder's weights
        self.init_network_weights_from_pretrain()

    def init_network_weights_from_pretrain(self):
        """

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

        """
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_path)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """

        """
        assert load_ae and (self.ae_net is not None), 'The trainer has not been initialized with an Autoencoder. It can thus not be loaded.'
        model_dict = torch.load(model_path, map_location=map_location)
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        if load_ae and (self.ae_net is not None):
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json_path):
        """

        """
        with open(export_json_path, 'w') as f:
            json.dump(self.results, f)


    def save_ae_results(self, export_json_path):
        """

        """
        with open(export_json_path, 'w') as f:
            json.dump(self.ae_results, f)
