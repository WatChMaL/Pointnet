# Python standard imports
from sys import stdout
from math import floor
from time import strftime, localtime
import numpy as np
import os

# PyTorch imports
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Custom imports
from io_util.sampler import SubsetSequentialSampler
from training_utils.engine import Engine
from training_utils.optimizer import select_optimizer
from training_utils.logger import CSVData

class EnginePointnet(Engine):

    def __init__(self, model, config):
        super().__init__(model, config)
        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=select_optimizer(config.optimizer, self.model_accs.parameters(),
                        **config.optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(self.optimizer, **config.scheduler_kwargs)
        self.keys = ['iteration', 'epoch', 'loss', 'acc']
        print(type(self.optimizer))

    def forward(self, data, mode="train"):
        """Overrides the forward abstract method in Engine.py.

        Args:
        mode -- One of 'train', 'validation'
        """

        # Set the correct grad_mode given the mode
        if mode == "train":
            self.model.train()
        elif mode in ["validation"]:
            self.model.eval()

        return self.model(data)

    def train(self):
        """Overrides the train method in Engine.py.

        Args: None
        """

        epochs          = self.config.epochs
        report_interval = self.config.report_interval
        valid_interval  = self.config.valid_interval
        num_val_batches = self.config.num_val_batches
        scheduler_step  = self.config.scheduler_step

        # Initialize counters
        epoch=0.
        iteration=0

        # Parameter to upadte when saving the best model
        best_val_loss=1000000.
        avg_val_loss=1000.
        next_scheduler = scheduler_step

        val_iter = iter(self.val_loader)

        # Global training loop for multiple epochs
        while (floor(epoch) < epochs):

            print('Epoch', np.round(epoch).astype(np.int),
                  'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
            train_iter = iter(self.train_loader)
            
            # Local training loop for a single epoch
            for idx, batch in enumerate(self.train_loader):
                data, label = batch

                # Update the epoch and iteration
                epoch+=1. / len(self.train_loader)
                iteration += 1

                # Do a forward pass
                pred = self.forward(data, mode="train")

                # Do a backward pass
                loss = self.backward(pred, label.view(-1))

                # Calculate metrics
                predlabel = torch.argmax(pred, dim=1)
                acc = torch.mean((predlabel == label.view(-1)).float())

                # Record the metrics for the mini-batch in the log
                self.train_log.record(self.keys, [iteration, epoch, loss, acc])
                self.train_log.write()

                # Print the metrics at report_intervals
                if iteration % report_interval == 0:
                    print("... Iteration %d ... Epoch %1.2f ... Loss %1.3f ... Acc %1.3f"
                          % (iteration, epoch, loss, acc))

                # Run validation on valid_intervals
                if iteration % valid_interval == 0:
                    print("Validating...")
                    val_loss=0.
                    val_acc=0.
                    with torch.no_grad():
                        for val_batch in range(num_val_batches):
                            try:
                                val_data=next(val_iter)
                            except StopIteration:
                                val_iter=iter(self.val_loader)
                                val_data=next(val_iter)
                            data, label = val_data

                            # Extract the event data from the input data tuple
                            pred = self.forward(data, mode="validation")
                            predlabel = torch.argmax(pred, dim=1)
                            acc = torch.mean((predlabel == label.view(-1)).float())
                            loss = self.criterion(pred, label.view(-1))
                            val_loss += loss.item()
                            val_acc += acc.detach().cpu().item()

                    val_loss /= num_val_batches
                    val_acc /= num_val_batches
                    if self.config.use_scheduler:
                        if iteration > next_scheduler:
                            self.scheduler.step(val_loss)
                            next_scheduler += scheduler_step


                    # Record the validation stats to the csv
                    self.val_log.record(self.keys, [iteration, epoch, val_loss, val_acc])
                    self.val_log.write()

                    # Save the best model
                    if val_loss < avg_val_loss:
                        self.save_state(mode="best", name="{}_{}".format(iteration, val_loss))
                        best_val_loss = val_loss
                        avg_val_loss = (val_loss * avg_val_loss)**0.5

                    # Save the latest model
                    self.save_state(mode="latest")
                    print("... Current Validation Loss %1.3f ... Best Validation Loss %1.3f"
                          % (val_loss, best_val_loss))

            self.save_state(mode="latest", name="epoch_{}".format(np.round(epoch).astype(np.int)))

        self.val_log.close()
        self.train_log.close()

    def validate(self, subset, name="current"):
        """Overrides the validate method in Engine.py.

        Args:
        subset          -- One of 'train', 'validation', 'test' to select the subset to perform validation on
        """
        # Print start message
        if subset == "train":
            message="Validating model on the train set"
        elif subset == "validation":
            message="Validating model on the validation set"
        else:
            print("validate() : arg subset has to be one of train, validation, test")
            return None

        print(message)

        # Setup the path to save output
        # Setup indices to use
        if subset == "train":
            self.log=CSVData(os.path.join(self.dirpath,"train_validation_log_{}.csv".format(name)))
            output_path=os.path.join(self.dirpath, "train_validation_{}".format(name))
            validate_indices = self.dataset.train_indices
        else:
            self.log=CSVData(os.path.join(self.dirpath,"valid_validation_log_{}.csv".format(name)))
            output_path=os.path.join(self.dirpath, "valid_validation_{}".format(name))
            validate_indices = self.dataset.val_indices

        os.makedirs(output_path)
        data_iter = DataLoader(self.dataset, batch_size=self.config.validate_batch_size,
                               num_workers=self.config.num_data_workers,
                               pin_memory=False, sampler=SubsetSequentialSampler(validate_indices))

        key_val = {"index":[], "label":[], "pred":[], "pred_val":[]}

        avg_loss = 0
        avg_acc = 0
        indices_iter = iter(validate_indices)

        dump_interval = self.config.validate_dump_interval
        dump_index = 0
        num_batches = 0

        with torch.no_grad():
            for iteration, batch in enumerate(data_iter):
                data, label = batch
                num_batches += 1

                #stdout.write("Iteration : {}, Progress {} \n".format(iteration, iteration/len(data_iter)))
                pred=self.forward(data, mode="validation")
                predlabel = torch.argmax(pred, dim=1)
                acc = np.mean(predlabel.cpu().numpy() == label.view(-1).cpu().numpy())
                loss = self.criterion(pred, label.view(-1)).item()
                avg_loss += loss
                avg_acc += acc

                # Log/Report
                self.log.record(["Iteration", "loss", "acc"], [iteration, loss, acc])
                self.log.write()

                # Log/Report
                for label, pred, preds in zip(label.tolist(), pred.argmax(1).tolist(), pred.exp().tolist()):
                    key_val["index"].append(next(indices_iter))
                    key_val["label"].append(label)
                    key_val["pred"].append(pred)
                    key_val["pred_val"].append(preds)

                # Check if iteration is valid_dump_interval
                if len(key_val["index"]) >= dump_interval:
                    #print("dumping")
                    name = os.path.join(output_path, "{}.npz".format(dump_index))
                    np.savez(name, **key_val)

                    dump_index += 1
                    key_val = {key:[] for key in key_val}

        self.log.close()

        name = os.path.join(output_path, "{}.npz".format(dump_index))
        np.savez(name, **key_val)

        avg_acc/=num_batches
        avg_loss/=num_batches

        stdout.write("Overall acc : {}, Overall loss : {}\n".format(avg_acc, avg_loss))

