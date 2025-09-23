'''
Module that contains the Base class for training and testing models locally and distributed.
'''
import os
import time
from typing import List, Dict, Any

#import mlflow
from torchvision import transforms
from s3torchconnector import S3Checkpoint
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import data_utilities as du

class RunMetrics:
    '''
    Class to hold various metrics captured during training runs and testing runs.
    '''
    def __init__(self):
        self.run_time = 0
        self.load_time = 0
        self.dataset_size = 0
        self.batches = 0
        self.testing_time = 0
        self.training_time = 0
        self.compute_time = 0 
        self.io_time = 0
        self.device_transfer_time = 0
        self.correct_predictions = 0 
        self.total_predictions = 0

    def accuracy(self) -> float:
        return self.correct_predictions / self.total_predictions


class TrainingBase:
    '''
    Base class for training and testing models locally and distributed.
    '''
    def __init__(self, parameters: Dict[str, Any]):
        self.logger = du.create_logger()
        self.parameters = parameters
        self.training_metrics = RunMetrics()
        self.testing_metrics = RunMetrics()

        if parameters['checkpoint']:
            self.checkpoint_uri = f"s3://{parameters['checkpoint_bucket']}/{parameters['project_name']} \
                                /{parameters['experiment_name']}/{parameters['run_name']} \
                                /{parameters['model_name']}"
            self.s3_checkpoint = S3Checkpoint(region=os.environ['AWS_REGION'])

        device = torch.device('cpu')
        if parameters['use_gpu']:
            if torch.xpu.is_available(): device = torch.device('xpu')
            if torch.cuda.is_available(): device = torch.device('cuda')
        self.logger.info(f'PyTorch Version: {torch.__version__}')
        self.logger.info(f'Using device: {device}')
        self.device = device

    def log_training_metrics(self):
        '''
        Method to log training metrics to the Python logger and to MLflow.
        '''
        self.logger.info(f'Total Training Run Time (in seconds) = {self.training_metrics.run_time:.4f}')
        self.logger.info(f'Load Time (in seconds) = {self.training_metrics.load_time:.4f}')
        self.logger.info(f'Dataset Size = {self.training_metrics.dataset_size}')
        self.logger.info(f'Batches = {self.training_metrics.batches}')
        self.logger.info(f'Total Training Time (in seconds) = {self.training_metrics.training_time:.4f}')
        self.logger.info(f'Compute Time (in seconds) = {self.training_metrics.compute_time:.4f}')
        self.logger.info(f'I/O Time (in seconds) = {self.training_metrics.io_time:.4f}')
        self.logger.info(f'Device Transfer Time (in seconds) = {(self.training_metrics.device_transfer_time):.4f}')
        utilization = (self.training_metrics.compute_time / self.training_metrics.training_time) * 100 if self.training_metrics.training_time > 0 else 0
        self.logger.info(f'Compute Utilization = {(utilization):.4f}%')

        metrics_dict = {
            'total_training_run_time': self.training_metrics.run_time,
            'training_load_time': self.training_metrics.load_time,
            'train_dataset_size': self.training_metrics.dataset_size,
            'batches': self.training_metrics.batches,
            'training_time': self.training_metrics.training_time,
            'training_compute_time': self.training_metrics.compute_time,
            'training_io_time': self.training_metrics.io_time,
            'training_device_transfer_time': self.training_metrics.device_transfer_time,
        }
        #mlflow.log_metrics(metrics_dict)

    def log_testing_metrics(self):
        '''
        Method to log testing metrics to the Python logger and to MLflow.
        '''
        incorrect = self.testing_metrics.total_predictions - self.testing_metrics.correct_predictions
        self.logger.info(f'Total Testing Run Time (in seconds) = {self.testing_metrics.run_time:.4f}')
        self.logger.info(f'Load Time (in seconds) = {self.testing_metrics.load_time:.4f}')
        self.logger.info(f'Dataset Size = {self.testing_metrics.dataset_size}')
        self.logger.info(f'Batches = {self.testing_metrics.batches}')
        self.logger.info(f'Total Test Time (in seconds) = {self.testing_metrics.testing_time:.4f}')
        self.logger.info(f'Compute Time (in seconds) = {self.testing_metrics.compute_time:.4f}')
        self.logger.info(f'I/O Time (in seconds) = {self.testing_metrics.io_time:.4f}')
        self.logger.info(f'Device Transfer Time (in seconds) = {(self.testing_metrics.device_transfer_time):.4f}')
        self.logger.info(f'Total Predictions: {self.testing_metrics.total_predictions}.')
        self.logger.info(f'Correct Predictions: {self.testing_metrics.correct_predictions}.')
        self.logger.info(f'Incorrect Predictions: {incorrect}.')
        self.logger.info(f'Accuracy: {self.testing_metrics.accuracy():.4f}.')

        metrics_dict = {
            'total_testing_run_time': self.testing_metrics.run_time,
            'testing_load_time': self.testing_metrics.load_time,
            'test_dataset_size': self.testing_metrics.dataset_size,
            'batches': self.testing_metrics.batches,
            'testing_time': self.testing_metrics.testing_time,
            'testing_compute_time': self.testing_metrics.compute_time,
            'testing_io_time': self.testing_metrics.io_time,
            'testing_device_transfer_time': self.testing_metrics.device_transfer_time,
            'total_predictions': self.testing_metrics.total_predictions,
            'correct_predictions': self.testing_metrics.device_transfer_time,
            'incorrect_predictions': incorrect,
            'accuracy': self.testing_metrics.accuracy(),
        }
        #mlflow.log_metrics(metrics_dict)

    def log_model(self, model: nn.Module, model_artifact_path: str, loader: DataLoader) -> None:
        '''
        Method to log the model to the Python logger and to MLflow.
        '''
        images, labels = next(iter(loader))
        image = images.to(self.device)
        label = labels.to(self.device)
        with torch.no_grad():
            output = model(image)

        #model_signature = mlflow.models.infer_signature(image.detach().to('cpu').numpy(), output.detach().to('cpu').numpy())
        #mlflow.pytorch.log_model(model, artifact_path=model_artifact_path, signature=model_signature)
        #self.logger.info(f'{model_artifact_path} has been logged to MLflow.')
        self.logger.info(f'Input size: {image.size()}')
        self.logger.info(f'Label size: {label.size()}')
        self.logger.info(f'Output size: {output.size()}')

    def train_model_dist(self, model: nn.Module, loader: DataLoader, parameters: Dict[str, Any]) -> None:
        '''
        Trains a model distributed with the specified data loader.
        '''

    def train_model_local(self, model: nn.Module, loader: DataLoader, parameters: Dict[str, Any]) -> None:
        '''
        Trains a model locally with the specified data loader.
        '''

    def run_local_training(self, parameters: Dict[str, Any], loader_type: str) -> None:
        '''
        Sets up the model and the data loader for local training against a training set.
        '''

    def checkpoint_model(self, model: nn.Module) -> None:
        '''
        Checkpoints the model using the previosly set up s3 checkpointer.
        '''
        if not self.parameters['checkpoint']:
            raise ValueError('Checkpointing is not enabled.')

        with self.s3_checkpoint.writer(self.checkpoint_uri) as writer:
            torch.save(model.state_dict(), writer)

    def test_model_local(self, loader: DataLoader, model: nn.Module) -> None:
        '''
        Test model locally.
        '''
        testing_start_time = time.perf_counter()

        model.to(self.device)
        model.eval()
        #model.eval().cuda()
        self.logger.info(f'Model moved to device: {self.device}')
        with torch.no_grad():
            # Batch loop
            batch_count:int = 0
            io_start: float = time.perf_counter()
            for x, y in tqdm(loader):
                # IO time for the batch.
                self.testing_metrics.io_time += time.perf_counter() - io_start
                # Iterable datasets do not have a __len__ method since batches are determined dynamically.
                batch_count += 1

                # Move to the specified device.
                device_start = time.perf_counter()
                x, y = x.to(self.device), y.to(self.device)
                self.testing_metrics.device_transfer_time += time.perf_counter() - device_start

                # Start of compute time for the batch - get the prediction and calculate accuracy.
                compute_start = time.perf_counter()

                y_pred = model(x)
                self.testing_metrics.correct_predictions += (y_pred.argmax(axis=1) == y).sum().item()
                self.testing_metrics.total_predictions += len(y)
                self.testing_metrics.compute_time += time.perf_counter() - compute_start

                # Need to set this here for the next loop.
                io_start = time.perf_counter()

        self.testing_metrics.batches = batch_count
        self.testing_metrics.testing_time = time.perf_counter() - testing_start_time

    def run_local_testing(self, parameters: Dict[str, Any], loader_type: str, model: nn.Module, data_transforms: transforms.Compose, start_new_experiment: bool=True)  -> None:
        '''
        Sets up the model and the data loader for testing the model locally against the test set.
        '''
