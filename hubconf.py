dependencies = ['torch']
import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from torch import optim
from torch.autograd import Variable
from collections import OrderedDict
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_tensor_to_pil = ToPILImage()
transform_pil_to_tensor = ToTensor()

# input_size = 784 # 28x28
input_size = 2 # Input size is determined later based on dataset and model
hidden_size = 500 
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001


class ModifiedDataset(Dataset):
  def __init__(self,given_dataset,shrink_percent=10):
    self.given_dataset = given_dataset
    self.shrink_percent = shrink_percent
    
  def __len__(self):
    return len(self.given_dataset)

  def __getitem__(self,idx):
    img, lab = self.given_dataset[idx]

    # print (type(img))
    # print (img.shape)

    img2 = transform_tensor_to_pil(img.squeeze())

    # print (img2.size)
    
    new_w = int(img2.size[0]*(1-self.shrink_percent/100.0))
    new_h = int(img2.size[1]*(1-self.shrink_percent/100.0))

    # print (new_w, new_h)

    img3 = img2.resize((new_w,new_h))

    # print (img3.size)

    x = transform_pil_to_tensor(img3)

    # print (x.shape)

    return x,lab

def get_dataloaders():
    # Import FashionMNIST dataset 
    train_data = torchvision.datasets.FashionMNIST(root='./data', 
                                            train=True, 
                                        transform=transforms.ToTensor(),  
                                            download=True)
    test_data = torchvision.datasets.FashionMNIST(root='./data', 
                                            train=False, 
                                            transform=transforms.ToTensor()) 


    mod_train_data = ModifiedDataset(train_data)
    mod_test_data = ModifiedDataset(test_data)

    print (train_data[0][0].shape)
    print (mod_train_data[0][0].shape)

    train_dataloader = DataLoader(mod_train_data, batch_size=batch_size)
    test_dataloader = DataLoader(mod_test_data, batch_size=batch_size)

    for x, y in test_dataloader:
        print(f"Shape of x [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader

class ConfigNeuralNet(nn.Module):
    def __init__(self, config, fc_size, num_classes):
        super(ConfigNeuralNet, self).__init__()
        self.layers = nn.Sequential(config)
        # self.layers = nn.ModuleList()
        # for (in_channels, out_channels, kernel_size, stride, padding) in config:
        #   self.layers.append(nn.Conv2d(
        #         in_channels=in_channels,              
        #         out_channels=out_channels,            
        #         kernel_size=kernel_size,              
        #         stride=stride,                   
        #         padding=padding,                  
        #     ),     
        #     )
        self.fc = nn.Linear(fc_size, num_classes)
        self.softmax = nn.Softmax(dim=1) 

    def return_dims(self,x):
        x = self.layers(x)
        return x.shape

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        out = self.softmax(x)
        # print(out.shape)
        return out

# My Cross Entropy Function
def softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x/sum_x

def log_softmax(x):
    return torch.log(softmax(x))

def my_loss(output, target):
    num_examples = target.shape[0]
    batch_size = outputs.shape[0]
    output = log_softmax(output)
    output = output[range(batch_size), target]
    return - torch.sum(output)/num_examples


# Function to determine input size
def determine_size(data_loader):
    fc_size = 20000
    model = ConfigNeuralNet(config, fc_size, num_classes).to(device)

    for x, _ in test_dataloader:
      data_point = x[0].to(device)
      dims = model.return_dims(data_point)
      print(dims)
      print(dims[0]*dims[1]*dims[2])
      return dims[0]*dims[1]*dims[2]

def train(model, train_dataloader, optimizer, criterion, num_epochs):
    n_total_steps = len(train_dataloader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):  
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            # images = images.reshape(-1,28*28).to(device)
            # images = images.reshape(1,1,100,28*28).to(device)
            # print(images.shape)
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}') 


def test(model, test_dataloader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_dataloader:
            # images = images.reshape(-1, 28*28).to(device)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item() 
            y_true = torch.Tensor.cpu(labels)
            y_pred = torch.Tensor.cpu(predicted)
            # print(y_true, y_pred)
            
        acc = 100.0 * n_correct / n_samples
        print(classification_report(y_true, y_pred))
        print(f'Accuracy of the network on the 10000 test images: {acc} %') 


def evaluate():
    train_dataloader, test_dataloader = get_dataloaders()

    config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')]

    ordered_dict = OrderedDict()
    index = 1

    for (in_channels, out_channels, kernel_size, stride, padding) in config:
        name = 'conv'+str(index)
        index = index+1
        ordered_dict[name] = nn.Conv2d(
            in_channels=in_channels,              
            out_channels=out_channels,            
            kernel_size=kernel_size,              
            stride=stride,                   
            padding=padding,                  
        )
    print(ordered_dict)
    config = ordered_dict
    fc_size = determine_size(test_dataloader)
    model = ConfigNeuralNet(config, fc_size, num_classes).to(device)
    # model = NeuralNet(20000, num_classes).to(device)
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = my_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    print(model)
    train(model, train_dataloader, optimizer, criterion, num_epochs)
    test(model, test_dataloader)




# """
# PLease Note: The main concept that I tried to use was gto pass a sample through the forward funciton, return the size and then use that 
# to create the fully connected layer. I think the concept is there but due to lack of time I have not been able to smoothen out the typos 
# and making it match so that it runs end to end. Kindly read the code manually.
# """
# # Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
# # This function is used to programatically identify the size of the fully connected layer and return it
# class test_CNN(nn.Module):
#     def __init__(self):
#         super(test_CNN, self).__init__()
#         self.conv1 = nn.Sequential(         
#             nn.Conv2d(
#                 in_channels=1,              
#                 out_channels=16,            
#                 kernel_size=5,              
#                 stride=1,                   
#                 padding=2,                  
#             ),                              
#             nn.ReLU(),                      
#             nn.MaxPool2d(kernel_size=2),    
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(16, 32, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )
#         # fully connected layer, output 10 classes
#         # self.out = nn.Linear(32 * 7 * 7, 10)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
#         x = x.view(x.size(0), -1)
#         print("PRINTING THE SIZE")
#         print(x.shape) 
#         return x.shape[1]      
#         # output = self.out(x)
#         # return output, x    # return x for visualization

# class cs19b012_CNN(nn.Module):
#     def __init__(self):
#         super(cs19b012_CNN, self).__init__()
#         self.conv1 = nn.Sequential(         
#             nn.Conv2d(
#                 in_channels=1,              
#                 out_channels=16,            
#                 kernel_size=5,              
#                 stride=1,                   
#                 padding=2,                  
#             ),                              
#             nn.ReLU(),                      
#             nn.MaxPool2d(kernel_size=2),    
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(16, 32, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )
#         # fully connected layer, output 10 classes
#         self.out = nn.Linear(32 * 7 * 7, 10)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
#         x = x.view(x.size(0), -1)       
#         output = self.out(x)
#         return output, x    # return x for visualization

# class cs19b012_CNN(nn.Module):
#     def __init__(self, sample_data):
#         super(cs19b012_CNN, self).__init__()
#         self.conv1 = nn.Sequential(         
#             nn.Conv2d(
#                 in_channels=1,              
#                 out_channels=16,            
#                 kernel_size=5,              
#                 stride=1,                   
#                 padding=2,                  
#             ),                              
#             nn.ReLU(),                      
#             nn.MaxPool2d(kernel_size=2),    
#         )
#         self.conv2 = nn.Sequential(         
#             nn.Conv2d(16, 32, 5, 1, 2),     
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                
#         )
#         # fully connected layer, output 10 classes
#         test_model = test_CNN()
#         fc_size = test_model.forward(sample_data)
#         # print(sample_data)
#         self.out = nn.Linear(32 * 7 * 7, 10)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
#         x = x.view(x.size(0), -1)
#         # print("PRINTING THE SIZE")
#         # print(x.shape)       
#         output = self.out(x)
#         return output, x    # return x for visualization

# def train(cnn, loss_func, optimizer, train_data_loader, num_epochs):
#     # num_epochs = 10
#     cnn.train()
        
#     # Train the model
#     total_step = len(train_data_loader)
        
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(train_data_loader):
            
#             # gives batch data, normalize x when iterate train_loader
#             b_x = Variable(images)   # batch x
#             b_y = Variable(labels)   # batch y
#             output = cnn(b_x)[0]               
#             loss = loss_func(output, b_y)
            
#             # clear gradients for this training step   
#             optimizer.zero_grad()           
            
#             # backpropagation, compute gradients 
#             loss.backward()    
#             # apply gradients             
#             optimizer.step()                
            
#             if (i+1) % 100 == 0:
#                 print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
#                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#             pass
        
#         pass
    
    
#     pass
#     # PATH = './saved_models/FMNIST_model.pth'
#     # torch.save(cnn.state_dict(), PATH)

# # sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
# def get_model(train_data_loader=None, n_epochs=10):
#     sample_data = None
#     for test_images, test_labels in train_data_loader:  
#         sample_image = test_images[0]    # Reshape them according to your needs.
#         sample_data = sample_image
    
#     model = cs19b012_CNN(sample_data)
#     loss_func = nn.CrossEntropyLoss()   
#     optimizer = optim.Adam(model.parameters(), lr = 0.01)   
#     # return cnn, loss_func, optimizer
#     train(model, loss_func, optimizer, train_data_loader, n_epochs)


#   # Use softmax and cross entropy loss functions
#   # set model variable to proper object, make use of train_data
  
#     print ('Returning model... (rollnumber: cs19b012)')
  
#     return model

# # sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)

# class MyModule(nn.Module):
#     def __init__(self, config, sample_data):
#         super(MyModule, self).__init__()
#         self.layers = nn.ModuleList()
#         for (in_channels, out_channels, kernel_size, stride, padding) in config:
#           self.layers.append(nn.Conv2d(
#                 in_channels=in_channels,              
#                 out_channels=out_channels,            
#                 kernel_size=kernel_size,              
#                 stride=stride,                   
#                 padding=padding,                  
#             ),     
#             )
#         # self.linears = nn.ModuleList([nn.Linear(10, 20) for _ in range(10)])
#         self.linear = nn.Linear(self.get_fc_size(sample_data),10)

#     def get_fc_size(self, x):
#         x = self.layers(x)
#         x = x.view(x.size(0), -1)
#         return x.shape

#     def forward(self, x, indices):
#         x = self.layers[indices](x) 
#         x = x.view(x.size(0), -1)
#         print("print the size", x.shape)
#         x = self.linear(x)
#         return x

# def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
#     for test_images, test_labels in train_data_loader:  
#         sample_image = test_images[0]    # Reshape them according to your needs.
#         sample_data = sample_image
#         break
#     model = MyModule(config, sample_data)    
#     print ('Returning model... (rollnumber: cs19b012)')
#     return model

#     # sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
# def test_model(model1=None, test_data_loader=None):

#     accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
#     for image, label in test_data_loader:
#         image = Variable(image)
#         label = Variable(label)
#         output = model1(image)
#         _, predicted = torch.max(output.data, 1)
#         accuracy_val += (predicted == label).sum().item()
#         precision_val += precision_score(label, predicted, average='macro')
#         recall_val += recall_score(label, predicted, average='macro')
#         f1score_val += f1_score(label, predicted, average='macro')
#     # precision_val, recall_val = precision_recall(preds, target, average='macro', num_classes=3)
    
#     print ('Returning metrics... (rollnumber: cs19b012)')
    
#     return accuracy_val, precision_val, recall_val, f1score_val