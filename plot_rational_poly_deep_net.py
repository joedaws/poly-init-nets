# script to generate plots for the NIPS 2019 paper
# uses DeepNet class
# approximates a periodic function
# generates plots for Figure 4
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from deep import DeepNet

# instantiate the network
a = -1
b = 1
L = 7
net = DeepNet(L,1,a,b)
# this network is for uninitialized
# case
unnet = DeepNet(L,1,a,b)
unnet.xavier_init()

# make some input data
N = 100
points = torch.FloatTensor(N,1).uniform_(a,b)
x = points.reshape(N,1)
y = net(x)

# sample a function of interest 
xx = points.numpy()
ly = 1./(1.+20*(0.5-xx)**2)

# create tensor target from data
target = torch.tensor(ly).type(y.type())
target = target.reshape(N,1)

# evaluate randomly initialized network
pN = 1001
px = torch.linspace(-1,1,pN)
px = px.reshape(pN,1)
p_rand_init_y = unnet(px)
ply = 1./(1.+20*(0.5-px)**2)

# initialize network with polynomial parameters
net.poly_init()
p_poly_init_y = net(px)

############
#TRAINING 1#
############

# Define loss as L2
criterion = torch.nn.MSELoss()
# We will use Adam optimizer
optimizer = optim.Adam(net.parameters(),lr=5e-3,weight_decay=0)

# define number of epochs to take
eN = 2000
y_pred = net(x)
loss = criterion(y_pred, target)
print('\n Initial loss')
print(loss.item())
print('')
print('--Begin Training--Stage 1--')
print(eN,' epochs will be used')

losses = []
for epoch in range(1,eN):
    y_pred = net(x)
    loss = criterion(y_pred, target)
    if epoch % 100 == 0:
        print ("epoch #", epoch)
        print (loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# second stage training
eN2 = 3000
optimizer = optim.Adam(net.parameters(),lr=1e-5,weight_decay=0)
print('--Begin Training--Stage 2--')
for epoch in range(eN,eN+eN2):
    y_pred = net(x)
    loss = criterion(y_pred, target)
    if epoch % 100 == 0:
        print ("epoch #", epoch)
        print (loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

p_trained_y = net(px)
# save learned parameters of the network
save_str = './data/rational_poly_deep_net.pt'
torch.save(net.state_dict(),save_str)

############
#TRAINING 2#
############

# Define loss as L2
criterion = torch.nn.MSELoss()
# We will use Adam optimizer
optimizer = optim.Adam(unnet.parameters(),lr=5e-3,weight_decay=0)

# define number of epochs to take
eN = 2000
y_pred = unnet(x)
loss = criterion(y_pred, target)
print('\n Initial loss')
print(loss.item())
print('')
print('--Begin Training--Stage 1--')
print(eN,' epochs will be used')

unlosses = []
for epoch in range(1,eN):
    y_pred = unnet(x)
    loss = criterion(y_pred, target)
    if epoch % 100 == 0:
        print ("epoch #", epoch)
        print (loss.item())
    unlosses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# second stage training
eN2 = 3000
optimizer = optim.Adam(unnet.parameters(),lr=1e-5,weight_decay=0)
print('--Begin Training--Stage 2--')
for epoch in range(eN,eN+eN2):
    y_pred = unnet(x)
    loss = criterion(y_pred, target)
    if epoch % 100 == 0:
        print ("epoch #", epoch)
        print (loss.item())
    unlosses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

p_unnet_trained_y = unnet(px)

# save learned parameters of the network
save_str = './data/rational_poly_deep_unnet.pt'
torch.save(unnet.state_dict(),save_str)

##########
#PLOTTING#
##########

# plot losses
lossfig = plt.figure(0)
plt.plot(losses,label='Polynomial Initialized')
plt.plot(unlosses,label='Xavier Initialized')
plt.legend()
plt.title('Training Losses')
plt.xlabel('Epoch Number')
plt.savefig('./fig/rational_poly_deep_net_both_training_losses.png')

# plot randomly initialized network
fig0 = plt.figure(1)
ax0 = fig0.add_subplot(1,1,1)
ax0.plot(px.numpy(),p_poly_init_y.detach().numpy(),label='Polynomial Initialized')
ax0.plot(px.numpy(),p_rand_init_y.detach().numpy(),label='Xavier Initialized')
ax0.plot(px.numpy(),ply.numpy(),label='Target')
ax0.set_title('Initialized Networks and Target')
plt.legend()
plt.savefig('./fig/rational_poly_deep_net_both_init.png')

# plot trained network
fig2 = plt.figure(3)
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(px.numpy(),p_trained_y.detach().numpy(),label='Polynomial Initialized')
ax2.plot(px.numpy(),p_unnet_trained_y.detach().numpy(),label='Xavier Initialized')
ax2.scatter(x.numpy(),ly,label='Training Data')
ax2.set_title('Trained Networks')
plt.legend()
plt.savefig('./fig/rational_poly_deep_net_both_trained_net.png')

