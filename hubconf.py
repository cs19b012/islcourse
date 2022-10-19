dependencies = ['torch']

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
# from torchmetrics.functional import precision_recall
from sklearn.metrics import precision_score, recall_score, f1_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  

"""
PLease Note: The main concept that I tried to use was gto pass a sample through the forward funciton, return the size and then use that 
to create the fully connected layer. I think the concept is there but due to lack of time I have not been able to smoothen out the typos 
and making it match so that it runs end to end. Kindly read the code manually.
"""
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
# This function is used to programatically identify the size of the fully connected layer and return it
class test_CNN(nn.Module):
    def __init__(self):
        super(test_CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        # self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        print("PRINTING THE SIZE")
        print(x.shape) 
        return x.shape[1]      
        # output = self.out(x)
        # return output, x    # return x for visualization

class cs19b012_CNN(nn.Module):
    def __init__(self):
        super(cs19b012_CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

class cs19b012_CNN(nn.Module):
    def __init__(self, sample_data):
        super(cs19b012_CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        test_model = test_CNN()
        fc_size = test_model.forward(sample_data)
        # print(sample_data)
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        # print("PRINTING THE SIZE")
        # print(x.shape)       
        output = self.out(x)
        return output, x    # return x for visualization

def train(cnn, loss_func, optimizer, train_data_loader, num_epochs):
    # num_epochs = 10
    cnn.train()
        
    # Train the model
    total_step = len(train_data_loader)
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_data_loader):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass
        
        pass
    
    
    pass
    # PATH = './saved_models/FMNIST_model.pth'
    # torch.save(cnn.state_dict(), PATH)

# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
    sample_data = None
    for test_images, test_labels in train_data_loader:  
        sample_image = test_images[0]    # Reshape them according to your needs.
        sample_data = sample_image
    
    model = cs19b012_CNN(sample_data)
    loss_func = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr = 0.01)   
    # return cnn, loss_func, optimizer
    train(model, loss_func, optimizer, train_data_loader, n_epochs)


  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
    print ('Returning model... (rollnumber: cs19b012)')
  
    return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)

class MyModule(nn.Module):
    def __init__(self, config, sample_data):
        super(MyModule, self).__init__()
        self.layers = nn.ModuleList()
        for (in_channels, out_channels, kernel_size, stride, padding) in config:
          self.layers.append(nn.Conv2d(
                in_channels=in_channels,              
                out_channels=out_channels,            
                kernel_size=kernel_size,              
                stride=stride,                   
                padding=padding,                  
            ),     
            )
        # self.linears = nn.ModuleList([nn.Linear(10, 20) for _ in range(10)])
        self.linear = nn.Linear(self.get_fc_size(sample_data),10)

    def get_fc_size(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        return x.shape

    def forward(self, x, indices):
        x = self.layers[indices](x) 
        x = x.view(x.size(0), -1)
        print("print the size", x.shape)
        x = self.linear(x)
        return x

def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
    for test_images, test_labels in train_data_loader:  
        sample_image = test_images[0]    # Reshape them according to your needs.
        sample_data = sample_image
        break
    model = MyModule(config, sample_data)    
    print ('Returning model... (rollnumber: cs19b012)')
    return model

    # sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

    accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
    for image, label in test_data_loader:
        image = Variable(image)
        label = Variable(label)
        output = model1(image)
        _, predicted = torch.max(output.data, 1)
        accuracy_val += (predicted == label).sum().item()
        precision_val += precision_score(label, predicted, average='macro')
        recall_val += recall_score(label, predicted, average='macro')
        f1score_val += f1_score(label, predicted, average='macro')
    # precision_val, recall_val = precision_recall(preds, target, average='macro', num_classes=3)
    
    print ('Returning metrics... (rollnumber: cs19b012)')
    
    return accuracy_val, precision_val, recall_val, f1score_val