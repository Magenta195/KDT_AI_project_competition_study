'''
Contains functions for training and testing a Pytorch model.
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets

from torchmetrics import Accuracy

from tqdm.notebook import tqdm_notebook

from typing import List

from python_scripts import utils

def train_step_gradient_accumulation(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Accuracy,
    device: torch.device,
    accumulation_num: int = 2,
    metric_learning: bool = False,
  ):
  model.train()
  loss_accu = 0
  acc_accu = 0
  iter_total = 0
  optimizer.zero_grad()

  for X_batch_train, y_batch_train in tqdm_notebook(dataloader, desc='train', leave=False):
    X_batch_train, y_batch_train = X_batch_train.to(device), y_batch_train.to(device)

    if metric_learning:
      y_logits = model(X_batch_train, y_batch_train)
    else:
      y_logits = model(X_batch_train)
    y_pred = torch.argmax(y_logits, dim=1)

    loss = loss_fn(y_logits, y_batch_train) / accumulation_num
    loss_accu += loss * accumulation_num
    acc = accuracy_fn(y_pred.to('cpu'), y_batch_train.to('cpu'))
    acc_accu += acc

    loss.backward()

    if (iter_total + 1) % accumulation_num == 0:
      optimizer.step()
      optimizer.zero_grad()

    iter_total += 1

  optimizer.step()

  return loss_accu / len(dataloader), acc_accu / len(dataloader)

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn: Accuracy,
    device: torch.device,
    metric_learning: bool = False,
  ):
  model.eval()
  test_loss_accu = 0
  test_acc_accu = 0

  with torch.inference_mode():
    for X_batch_test, y_batch_test in tqdm_notebook(dataloader, desc='test', leave=False):
      X_batch_test, y_batch_test = X_batch_test.to(device), y_batch_test.to(device)

      if metric_learning:
        test_logits = model(X_batch_test, y_batch_test)
      else:
        test_logits = model(X_batch_test)
      test_pred = torch.argmax(test_logits, dim=1)

      test_loss = loss_fn(test_logits, y_batch_test)
      test_loss_accu += test_loss
      test_acc = accuracy_fn(test_pred.to('cpu'), y_batch_test.to('cpu'))
      test_acc_accu += test_acc

  return test_loss_accu / len(dataloader), test_acc_accu / len(dataloader)

def test_step_metric_learning(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    accuracy_fn: Accuracy,
    device: torch.device
  ):
  model.eval()
  test_acc_accu = 0
  train_features = []
  train_labels = []

  with torch.inference_mode():
    for X_batch_train, y_batch_train in tqdm_notebook(train_dataloader, desc='test_metric_1', leave=False):
      X_batch_train, y_batch_train = X_batch_train.to(device), y_batch_train.to(device)

      train_features.append(model.backbone(X_batch_train).detach().cpu())
      train_labels.append(y_batch_train.cpu())

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    for X_batch_test, y_batch_test in tqdm_notebook(test_dataloader, desc='test_metric_2', leave=False):
      X_batch_test, y_batch_test = X_batch_test.to(device), y_batch_test.to(device)

      test_features = model.backbone(X_batch_test).detach().cpu()
      test_preds = train_labels[torch.argmax(
         torch.mm(nn.functional.normalize(test_features), nn.functional.normalize(train_features).T),
         dim=1
      )]

      test_acc = accuracy_fn(test_preds.to('cpu'), y_batch_test.to('cpu'))
      test_acc_accu += test_acc

  return test_acc_accu / len(test_dataloader)

def train_tensorboard_gradient_accumulation(
      model: torch.nn.Module,
      save_name: str,
      train_dataloader: torch.utils.data.DataLoader,
      test_dataloader: torch.utils.data.DataLoader,
      loss_fn: torch.nn.Module,
      optimizer: torch.optim.Optimizer,
      accuracy_fn: Accuracy,
      epochs: int,
      device: torch.device,
      writer: SummaryWriter = None,
      accumulation_num: int = 2,
      saving_max: bool = False,
      metric_learning: bool = False,
):
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': [],
               'test_acc_metric': []}

    max_acc = 0

    model.to(device)

    for epoch in tqdm_notebook(range(epochs), desc=save_name, leave=True):
        train_loss, train_acc = train_step_gradient_accumulation(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy_fn=accuracy_fn,
                                           device=device,
                                           accumulation_num=accumulation_num,
                                           metric_learning=metric_learning)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device,
                                        metric_learning=metric_learning)

        results['train_loss'].append(train_loss.detach().cpu().numpy())
        results['train_acc'].append(train_acc.detach().cpu().numpy())
        results['test_loss'].append(test_loss.detach().cpu().numpy())
        results['test_acc'].append(test_acc.detach().cpu().numpy())

        print(f'Epoch: {epoch} | Train_loss: {train_loss:.4f}, Train_acc: {train_acc:.4f} | Test_loss: {test_loss:.4f}, Test_acc: {test_acc:.4f}')

        if metric_learning and epoch % 3 == 0:
          test_acc = test_step_metric_learning(
             model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             accuracy_fn=accuracy_fn,
             device=device
          )
          results['test_acc_metric'].append(test_acc.detach().cpu().numpy())
          print(f'Test_acc_metric: {test_acc:.4f}')

        if saving_max and test_acc > max_acc:
           max_acc = test_acc
           utils.save_model(
              model=model,
              target_dir='../models',
              model_name=f'{save_name}_EPOCH_{epoch}_TEST-ACC_{test_acc:.4f}.pth'
           )

        if writer:
          writer.add_scalars(main_tag='Loss',
                            tag_scalar_dict={'train_loss': results['train_loss'][-1],
                                              'test_loss': results['test_loss'][-1]},
                            global_step=epoch)
          writer.add_scalars(main_tag='Accuracy',
                            tag_scalar_dict={'train_acc': results['train_acc'][-1],
                                              'test_acc': results['test_acc'][-1]},
                            global_step=epoch)
        #   writer.add_graph(model=model,
        #                   input_to_model=torch.randn(32, 3, 224, 224).to(device))
          writer.close()

    return results

def HP_tune_train(
    model: torchvision.models,
    model_generator: callable,
    model_weights: torchvision.models,
    model_name: str,
    train_dataset: datasets,
    test_dataset: datasets,
    class_names: List[any],
    learning_rate_list: List[float],
    weight_decay_list: List[float],
    epochs_list: List[int],
    batch_size_list: List[int],
    is_tensorboard_writer: bool,
    device: torch.device,
    gradient_accumulation_num: int = 2,
    saving_max: bool = False,
    metric_learning: bool = False,
):
    tuning_results = []

    for learning_rate in learning_rate_list:
        for weight_decay in weight_decay_list:
            for epochs in epochs_list:
                for batch_size in batch_size_list:
                    t_result = {
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'epochs': epochs,
                        'batch_size': batch_size
                    }

                    train_dataloader = DataLoader(
                        dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                    )
                    test_dataloader = DataLoader(
                        dataset=test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                    )

                    if not model:
                        model = model_generator(weights=model_weights)

                    writer = None
                    if is_tensorboard_writer:
                        writer = utils.create_writer(
                            experiment_name= model_name + '_test',
                            model_name=model_name,
                            extra=f'LR_{learning_rate}_WD_{weight_decay}_EP_{epochs}_BS_{batch_size}_GA_{gradient_accumulation_num}'
                        )

                    model_results = train_tensorboard_gradient_accumulation(
                        model=model,
                        save_name=f'{model_name}_LR_{learning_rate}_WD_{weight_decay}_BS_{batch_size}_GA_{gradient_accumulation_num}',
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=nn.CrossEntropyLoss(),
                        optimizer=torch.optim.Adam(
                            params=model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay
                        ),
                        accuracy_fn=Accuracy(
                            task='multiclass',
                            num_classes=len(class_names)
                        ),
                        epochs=epochs,
                        device=device,
                        writer=writer,
                        accumulation_num=gradient_accumulation_num,
                        saving_max=saving_max,
                        metric_learning=metric_learning
                    )

                    t_result['train_loss'] = model_results['train_loss'][-1]
                    t_result['test_loss'] = model_results['test_loss'][-1]
                    t_result['train_acc'] = model_results['train_acc'][-1]
                    t_result['test_acc'] = model_results['test_acc'][-1]

                    tuning_results.append(t_result)

    return tuning_results
