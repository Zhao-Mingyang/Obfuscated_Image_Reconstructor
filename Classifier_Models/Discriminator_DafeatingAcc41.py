import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
       
        # max pooling layer
        self.pool1 = nn.MaxPool2d((2, 2),2)
        #linear layer (512 -> 2)
        self.fc1 = nn.Linear(32768,1024)
        self.fc3 = nn.Linear(1024, 200)
        self.LeakyReLU = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.30)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x =  self.pool1(self.LeakyReLU(self.conv1(x)))
        x = self.pool1(self.LeakyReLU(self.conv2(x)))
        x =  self.pool1(self.LeakyReLU(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 32768)

        
        # add 1st hidden layer, with relu activation function
        x = self.LeakyReLU(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    
# n_epochs = 30

# valid_loss_min = np.Inf # track change in validation loss

# for epoch in range(1, n_epochs+1):

#     # keep track of training and validation loss
#     train_loss = 0.0
#     valid_loss = 0.0
    
#     ###################
#     # train the model #
#     ###################
#     model.train()
#     for i_batch, sample_batched in enumerate(dataloader):
#         images_batch, landmarks_batch = \
#                 sample_batched['image'], sample_batched['landmarks']
#         data=images_batch.float()
#         target = landmarks_batch.long()
#         if train_on_gpu:
#             data, target = images_batch.cuda(), landmarks_batch.cuda()
#             data = data.float()
#             target = target.long()
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         output = model(data)


#         # calculate the batch loss
#         loss = criterion(output, target)
#         # backward pass: compute gradient of the loss with respect to model parameters
#         loss.backward()
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#         # update training loss
#         train_loss += loss.item()*data.size(0)
        
#     ######################    
#     # validate the model #
#     ######################
#     model.eval()
#     for i_batch, sample_batched in enumerate(dataloader_val):
#         images_batch, landmarks_batch = \
#                 sample_batched['image'], sample_batched['landmarks']
#         data=images_batch.float()
#         target = landmarks_batch.long()
#         if train_on_gpu:
#             data, target = images_batch.cuda(), landmarks_batch.cuda()
#             data = data.float()
#             target = target.long()
#         output = model(data)
#         # calculate the batch loss
#         loss = criterion(output, target)
#         # update average validation loss 
#         valid_loss += loss.item()*data.size(0)
    
#     # calculate average losses
#     train_loss = train_loss/len(dataloader)
#     valid_loss = valid_loss/len(dataloader_val)
        
#     # print training/validation statistics 
#     print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
#         epoch, train_loss, valid_loss))
    
#     # save model if validation loss has decreased
#     if valid_loss <= valid_loss_min:
#         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#         valid_loss_min,
#         valid_loss))
#         torch.save(model.state_dict(), 'model_cifar.pt')
#         valid_loss_min = valid_loss
