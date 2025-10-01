'''
This module drives the MNIST training options.
'''
import argparse
from multiprocessing import get_context
import os
import time
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
#import mlflow
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader

import data_utilities as du
from training_base import TrainingBase
import unet3d_loaders as ul


# Load the credentials and connection information.
load_dotenv('unet3d.env')
UNET3D_BUCKET_NAME = os.environ['UNET3D_BUCKET_NAME']
CHECKPOINT_BUCKET = os.environ['CHECKPOINT_BUCKET']
#COMPUTATION_TIME = os.environ['COMPUTATION_TIME']

class TrainUNET3D(TrainingBase):
    '''
    Class that contains training and testing methods for both local processing and distributed
    processing.
    '''

    def train_model_local(self, model: nn.Module, loader: DataLoader, parameters: Dict[str, Any]) -> None:
        '''
        Trains a model with the specified data loader.
        '''
        training_start_time = time.perf_counter()

        #model.to(self.device)
        #self.logger.info(f'Model moved to device: {self.device}')

        # Define the loss function and optimizer..
        #loss_func = nn.NLLLoss()
        #optimizer = optim.SGD(model.parameters(), lr=parameters['lr'], momentum=parameters['momentum'])

        # Epoch loop
        for epoch in range(parameters['epochs']):

            # Initialize epic based metrics.
            total_loss = 0
            epoch_compute_time: float = 0.0
            epoch_io_time: float = 0.0
            epoch_device_transfer_time: float = 0.0
            io_start: float = time.perf_counter()
            batch_count:int = 0
            epoch_byte_size:int = 0

            # Batch loop
            for samples in loader:
                # IO time for the batch.
                batch_io_time = time.perf_counter() - io_start
                # Iterable datasets do not have a __len__ method since batches are determined dynamically.
                batch_count += 1

                batch_byte_size: int = 0
                for sample in samples:
                    batch_byte_size += len(sample)

                #number_of_elements = samples.numel()
                #element_byte_size = samples.element_size()
                #batch_byte_size = number_of_elements * element_byte_size
                epoch_byte_size += batch_byte_size

                #device_start = time.perf_counter()
                #images, labels = images.to(self.device), labels.to(self.device)
                #batch_device_transfer_time = time.perf_counter() - device_start

                # Start of compute time for the batch.
                compute_start = time.perf_counter()
                time.sleep(float(parameters['computation_time']))  # Simulate GPU compute time.

                # Training pass
                #optimizer.zero_grad()
                #output = model(images)
                #loss = loss_func(output, labels)
                #loss.backward()
                #optimizer.step()

                # Loss calculations
                #total_loss += loss.item()

                # Track timing metrics.
                batch_compute_time = time.perf_counter() - compute_start
                epoch_compute_time += batch_compute_time
                #epoch_device_transfer_time += batch_device_transfer_time
                epoch_io_time += batch_io_time
                utilization = (batch_compute_time / (batch_compute_time+batch_io_time)) * 100
                self.logger.info(f'Epoch {epoch+1} - ' \
                            f'Batch {batch_count} - ' \
                            f'Batch Size (bytes): {(batch_byte_size/1e9):.4fGb} - ' \
                            f'Compute time: {batch_compute_time:.4f} - ' \
                            f'IO time: {batch_io_time:.4f} - ' \
                            #f'Device Transfer time: {batch_device_transfer_time:.4f} - ' \
                            f'Utilization: {utilization:.2f}%.')

                # Need to set this here for the next loop.
                io_start = time.perf_counter()

            self.logger.info(f'Epoch {epoch+1} - ' \
                             f'Compute time: {epoch_compute_time:.4f} - ' \
                             f'IO time: {epoch_io_time:.4f}.')

            io_bandwidth = ((epoch_byte_size/epoch_io_time) * 8) / 1e9
            compute_bandwidth = ((epoch_byte_size/epoch_compute_time) * 8) / 1e9
            self.logger.info(f'Epoch {epoch+1} - ' \
                             f'Compute bandwidth: {compute_bandwidth:.4f} Gbps - ' \
                             f'IO bandwidth: {io_bandwidth:.4f} Gbps.')

            self.training_metrics.batches = batch_count
            self.training_metrics.compute_time += epoch_compute_time
            self.training_metrics.io_time += epoch_io_time
            #self.training_metrics.device_transfer_time += epoch_device_transfer_time

            # Save checkpoint to S3 if enabled.
            if parameters['checkpoint']:
                self.checkpoint_model(model)

        self.training_metrics.training_time = time.perf_counter() - training_start_time

    def run_local_training(self, parameters: Dict[str, Any], loader_type: str):
        '''
        Set up the model and the data loader for local training.
        '''
        self.logger.info(f'Loader type: {loader_type}')
        run_start_time = time.perf_counter()

        # No model for now.
        model = None

        # Create the training loader.
        split = 'train'
        train_loader, load_time = ul.create_unet3d_loader(parameters['bucket_name'], split,
                                                            loader_type, parameters['batch_size'],
                                                            num_workers=parameters['num_workers'],
                                                            prefetch_factor=parameters['prefetch_factor'],
                                                            smoke_test_count=parameters['smoke_test_count'])
        self.logger.info(f'Training Loader Creation Time (in seconds) = {load_time:.4f}')
        self.training_metrics.load_time = load_time

        # Iterable datasets do not have a __len__ method since batches are determined dynamically.
        if not 'iter' in loader_type:
            self.training_metrics.dataset_size = len(train_loader.dataset)

        # Train the model.
        self.train_model_local(model, train_loader, parameters)
        self.training_metrics.run_time = time.perf_counter() - run_start_time
        self.log_training_metrics()


def single_process(bucket_name: str, split: str) -> float:
    '''
    Simple single threaded network test.
    '''
    # Get a list of objects for either train or val.
    X = du.get_unet3d_list(bucket_name, split, smoke_test_count=0)

    run_start_time = time.perf_counter()
    bandwidth = []
    for object_path in X:
        start = time.perf_counter()
        img = du.get_object_from_minio(bucket_name, object_path)
        io_time = time.perf_counter() - start
        byte_size = len(img)
        object_bandwidth = ((byte_size/io_time) * 8) / 1e9
        bandwidth.append(object_bandwidth)

    run_time = time.perf_counter() - run_start_time

    return run_time, bandwidth


def _worker(worker_id: int, samples: Sequence[str], out_q) -> None:
    '''
    Worker run in a subprocess: loops through each character of each string
    and reports elapsed time back via out_q as a tuple (idx, elapsed_seconds).
    '''
    worker_start = time.perf_counter()
    for object_path in samples:
        #start = time.perf_counter()
        img = du.get_object_from_minio(UNET3D_BUCKET_NAME, object_path)
        #print(f'Object Size: {(len(img)/1e9):.4f} Gb - IO Time: {time.perf_counter()-start:.4f}s')

    elapsed = time.perf_counter() - worker_start
    out_q.put((worker_id, elapsed))


def multi_process(object_list: List[str], num_workers: int) -> Dict[int, float]:
    '''
    Create 8 subprocesses and send each subprocess a list of strings.
    Each subprocess loops through the characters of each string and reports
    back how long its loop took.

    Args:
      lists_of_strings: list of lists of strings. If fewer than 8 lists are
                        provided the remaining processes receive empty lists.
                        If more than 8 are provided only the first 8 are used.

    Returns:
      Dict mapping process index (0..7) -> elapsed seconds.
    '''
    ctx = get_context('spawn')  # safe on macOS without requiring __main__ guard
    Queue = ctx.Queue
    Process = ctx.Process

    # Ensure exactly 8 tasks
    #tasks = lists_of_strings[:8] + [[] for _ in range(max(0, 8 - len(lists_of_strings)))]
    tasks = []
    total_length = len(object_list)
    objects_per_worker = total_length // num_workers

    for i in range(num_workers):
        tasks.append(object_list[i*objects_per_worker:(i+1)*objects_per_worker])

    q = Queue()
    procs = []
    start = time.perf_counter()
    for i, task in enumerate(tasks):
        p = Process(target=_worker, args=(i, task, q))
        p.start()
        procs.append(p)

    results: Dict[int, float] = {}
    # collect one result per started process
    for _ in procs:
        worker_id, elapsed = q.get()
        results[worker_id] = elapsed

    for p in procs:
        p.join()
    
    final_results = {i: results.get(i, 0.0) for i in range(num_workers)}
    wall_clock_time = time.perf_counter() - start

    return wall_clock_time, objects_per_worker, final_results


def main():
    '''
    Main function that processes the arguments sent to this module via the command line.
    '''
    parser = argparse.ArgumentParser(description='ML Command line interface.')
    parser.add_argument('-lb', '--list_buckets', help='List all buckets.', action='store_true')
    parser.add_argument('-lo', '--list_objects', help='List all objects in the specified bucket.')
    parser.add_argument('-eb', '--empty_bucket', help='Remove all objects in the specified bucket.')
    parser.add_argument('-load', '--load_bucket', help='Load the UNET3D dataset into the specified bucket.')
    parser.add_argument('-train', '--train', help='Train the UNET3D model.', action='store_true')
    parser.add_argument('-lt', '--loader_type', help='Type of loader to use for loading training and test sets ' \
                        '(map, iter, s3map or s3iter).')

    parser.add_argument('-sp', '--single_process', help='Single process test.')
    parser.add_argument('-mp', '--multi_process', help='Multi-process test.')

    args = parser.parse_args()

    if args.empty_bucket:
        count = du.empty_bucket(args.empty_bucket)
        print(f'Number of objects removed in {args.empty_bucket}:', count)
    if args.list_buckets:
        bucket_list = du.get_bucket_list()
        print(bucket_list)
    if args.list_objects:
        object_list = du.get_object_list(args.list_objects)
        print(f'Number of objects in {args.list_objects}:', len(object_list))
    if args.load_bucket:
        train_count, test_count = du.load_mnist_to_minio(args.load_bucket)
        print(f'MNIST training images added to {args.load_bucket}:', train_count)
        print(f'MNIST testing images added to {args.load_bucket}:', test_count)

    if args.single_process:
        run_time, bandwidth = single_process(args.single_process, 'train')
        print(f'Single Process Test (in seconds) = {run_time:.4f}')
        print(f'Max object bandwidth (in Gbps) = {max(bandwidth):.4f} Gbps')
        print(f'Avg object bandwidth (in Gbps) = {sum(bandwidth)/len(bandwidth):.4f} Gbps')
        print(f'Min object bandwidth (in Gbps) = {min(bandwidth):.4f} Gbps')

    if args.multi_process:
        # Prepare up to 8 lists of strings (fewer is OK; will be padded to 8)
        #object_list = ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh", "iiii", "jjjj","kkkk", "llll", "mmmm", "nnnn", "oooo", "pppp","qqqq", "rrrr", "ssss", "tttt","uuuu", "vvvv", "wwww", "xxxx","yyyy", "zzzz", "$$$"]
        object_list = du.get_unet3d_list(UNET3D_BUCKET_NAME, 'train', smoke_test_count=0)

        wall_clock_time, objects_per_worker, results = multi_process(object_list, int(args.multi_process))
        
        print(f'Objects per worker: {objects_per_worker}   Total objects: {len(object_list)}')
        for worker_id in sorted(results):
            print(f"Process {worker_id} elapsed: {results[worker_id]:.6f} s")
        print(f"Total wall-clock time to collect results: {wall_clock_time:.6f} s")

        #run_time = multi_process(args.multi_process, 'train', num_workers=8)
        #print(f'Multi Process Test (in seconds) = {run_time:.4f}')

    if args.train:
        # Hyperparameters
        model_name = 'unet3D'
        smoke_test_count = 0
        parameters = {
            'batch_size': 7,
            'bucket_name': UNET3D_BUCKET_NAME,
            'checkpoint': False,
            'checkpoint_bucket': CHECKPOINT_BUCKET,
            'computation_time': 0.636,
            'epochs': 1,
            'model_name': model_name,
            'num_workers': 32,
            'prefetch_factor': 4,
            'smoke_test_count': smoke_test_count,
            'use_gpu': True,
            }
        tm = TrainUNET3D(parameters)
        tm.run_local_training(parameters, args.loader_type)


if __name__ == '__main__':
    main()
