'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


class model():
    def __init__(self, depth=5):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
#         self.num_train_samples=0
#         self.num_feat=1
#         self.num_labels=1
        self.is_trained=False

        #self.depth=depth
        #self.learner = RandomForestClassifier(max_depth=self.depth)

        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        self.learner = model.to(self.device)

    def fit(self,dataloaders): # X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        num_epochs=1
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(self.learner.classifier[1].parameters(), lr=0.001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        #Todo. That breaks when the data changes a bit
        dataset_sizes = {'val': 2173, 'train': 8690}

        since = time.time()

        #best_model_wts = copy.deepcopy(model.state_dict())
        best_model_wts = copy.deepcopy(self.learner.state_dict())
        best_acc = 0.0
        loss_history = []
        score_history = []
        train_loss, train_score, valid_loss, valid_score = [], [], [], []

        for epoch in range(num_epochs):
            print('\n','Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.learner.train()  # Set model to training mode
                else:
                    self.learner.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.learner(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # end dataloader loop

                if phase == 'train':
                    scheduler.step()
#                     loss_history.append(running_loss)
#                     score_history.append(running_corrects)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                loss_history, score_history = (train_loss, train_score) if phase == 'train' else (valid_loss, valid_score)
                loss_history.append(epoch_loss)
                score_history.append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.learner.state_dict())
            # end phase loop

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        self.learner.load_state_dict(best_model_wts)

        #return dict(   
        #    model = model,
        #    train_loss = train_loss,
        #    train_score = train_score,
        #    valid_loss = valid_loss,
        #    valid_score = valid_score,
        #)


        self.is_trained=True
        return self.learner
        
    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
#         num_test_samples = X.shape[0]
#         if X.ndim>1: num_feat = X.shape[1]
#         print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
#         if (self.num_feat != num_feat):
#             print("ARRGH: number of features in X does not match training data!")
#         print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
#         y = np.zeros([num_test_samples, self.num_labels])
        # If you uncomment the next line, you get pretty good results for the Iris data :-)
        #y = np.round(X[:,3])
        self.learner
        y = self.learner.predict(X)
        return y

    def save(self, path="/MobileNet2"):
        #torch.save(self.learner['model'].state_dict(), path)
        path = "./sample_code_submission"+ path + "_save"
        torch.save(self.learner.state_dict(), path)
        

    def load(self, path="./"):
        self.learner.load_state_dict(torch.load(path,map_location=torch.device(self.device)))
        return self
        
