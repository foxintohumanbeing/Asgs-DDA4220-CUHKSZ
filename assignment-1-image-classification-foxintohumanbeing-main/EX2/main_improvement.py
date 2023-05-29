# warning supression
import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import os
import numpy as np

# This is a quite simple CNN with 3 convolutional layers
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # the first layer of the CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # the second layer of the CNN
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 7, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
          
            # GRADED FUNCTION: Please define the third layer of the CNN. 
            # Conv2D with 128 5x5 filters and stride of 2
            # ReLU
            # MaxPool2d with 2x2 filters and stride of 2
            ### START SOLUTION HERE ###
            nn.Conv2d(128,128,5,stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2)
            ### END SOLUTION HERE ###
        )
        self.classifier = nn.Sequential(
            # GRADED FUNCTION: Please define the classifier
            # Linear with input size of 128 x width? x height? and output size of 4096
            # ReLU
            # Linear with input size of 4096 and output size of 4096
            # ReLU
            # Linear with input size of 4096 and output size of number of classes
            ### START SOLUTION HERE ###
            nn.Linear(4608,4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace = True),
            nn.Linear(4096,4096),
            nn.Dropout(0.5),
            nn.ReLU(inplace = True),
            nn.Linear(4096,5)
            ### END SOLUTION HERE ###
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # GRADED FUNCTION: Flatten Layer
        ### START SOLUTION HERE ###
        x = torch.flatten(x,1)
        ### END SOLUTION HERE ###
        x = self.classifier(x)
        return x

def train(train_loader, model, loss_fn, optimizer, device):
    step_loss = []
    for i, (image, annotation) in enumerate(train_loader):
        # move data to the same device as model
        image = image.to(device)
        annotation = annotation.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward and compute prediction error
        output = model(image)
        loss = loss_fn(output, annotation)
        # backward + optimize
        loss.backward()
        optimizer.step()
        step_loss.append(loss.item())
        # print statistics
        if i % 20 == 0:    # print every 20 iterates
            print(f'iterate {i + 1}: loss={loss:>7f}')
    return np.array(step_loss).mean()

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    img = img.cpu()
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


def val(val_loader, model, device,epoch_num):
    # switch to evaluate mode
    model.eval()
    classes = ['daisy','dandelion','rose','sunflower','tulip']
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (image, annotation) in enumerate(val_loader):
            # move data to the same device as model
            image = image.to(device)
            annotation = annotation.to(device)
            # network forward
            output = model(image)
            # for compute accuracy
            _, predicted = torch.max(output.data, 1)
            total += annotation.size(0)
            if epoch_num > 190:
                if i == 1:
                    plt.rcParams["savefig.bbox"] = 'tight'
                    fig=plt.figure(figsize=(20,20))
                    for j in range(16):
                        plt.rc('font', size=9) 
                        plt.subplot(4,4,j+1)
                        plt.imshow(imageshow(image[j]))
                        plt.axis('off')
                        plt.title(f"label: {classes[annotation[j]]}")
                    plt.savefig(r'C:\\Users\\mayn\\Desktop\\assignment-1-image-classification-foxintohumanbeing\\pics\\epoch'+str(epoch_num)+'_improve.png')
            correct += (predicted == annotation).sum().item()

    # GRADED FUNCTION: calculate the accuracy using variables before
    # use variable named 'acc' to store the accuracy
    ### START SOLUTION HERE ###
    acc = correct/total
    ### END SOLUTION HERE ###
    print(f'total val accuracy: {100 * acc:>2f} %')
    return acc

if __name__ == '__main__':
    # define image transform
    transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                    ])
    batch_size = 64
    current_path = os.getcwd()
    # loda data
    traindir = './EX2/flower_dataset/train'
    valdir = './EX2/flower_dataset/val'
    # GRADED FUNCTION: define train_loader and val_loader
    ### START SOLUTION HERE ###
    train_set = datasets.ImageFolder(traindir,transform=transform)
    valid_set = datasets.ImageFolder(valdir,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valid_set,batch_size=batch_size, shuffle=False)
    ### END SOLUTION HERE ###

    # device used to train
    device = torch.device("cuda:0")
    # GRADED FUNCTION: define a SimpleCNN model and move it to the device
    # use variable named 'model' to store this model
    ### START SOLUTION HERE ###
    model = SimpleCNN()
    model = model.to(device)

    ### END SOLUTION HERE ###

    # Classification Cross-Entropy loss 
    loss_fn = nn.CrossEntropyLoss()

    # GRADED FUNCTION: Please define the optimizer as SGD with lr=0.05, momentum=0.9, weight_decay=0.0001
    ### START SOLUTION HERE ###
    ### START SOLUTION HERE ###
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9,weight_decay=0.0001)
    ### END SOLUTION HERE ###

    # GRADED FUNCTION: Please define the 
    ### START SOLUTION HERE ###
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma= 0.95,step_size = 5)

    # create model save path
    os.makedirs('work_dir', exist_ok=True)
    max_acc = -float('inf')
    validation_acc = []
    train_loss = []
    for epoch in range(200):
        print('-' * 30, 'epoch', epoch + 1, '-' * 30)

        # train
        loss = train(train_loader, model, loss_fn, optimizer, device)
        print('lr: {}'.format(optimizer.param_groups[0]['lr']))
        train_loss.append(loss)
        # validation
        acc = val(val_loader, model, device,epoch)
        # save best model
        if acc > max_acc:
            pt_path = os.path.join('EX2/work_dir', 'EX2_best_improved.pt')
            torch.save(model.state_dict(), pt_path)
            print('save model')
            max_acc = acc

        # decay learning rate
        scheduler.step()
        validation_acc.append(acc)

    fig,ax = plt.subplots()
    ax.plot(train_loss,color='red',marker = 'o',label = 'Training Loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Training Loss')
    ax2 = ax.twinx()
    ax2.plot(validation_acc,color='blue',marker = 'o',label = 'Validation Acc')
    ax2.set_ylabel('Validation Acc')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax2.set_title('Training Loss and Validation Accuracy')
    plt.savefig('C:/Users/mayn/Desktop/assignment-1-image-classification-foxintohumanbeing/pics/performance_improved.png')