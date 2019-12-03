import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim


class Input(nn.Module):
    def __init__(self):
        super(Input,self).__init__()
        self.fc1 = nn.Linear(10,1)
    def forward(self,data):
        res = self.fc1(data)
        return res



def main():
    input_test = Input()
    l2loss     = nn.MSELoss()
    optimizer = optim.Adam(input_test.parameters(), lr=0.01)
    print(input_test)
    for i in range(1,1000):
        input_test.zero_grad()
        a = torch.tensor(np.random.random((2,10))).float()
        a.requires_grad_(False)
        gt = torch.mean(a,1).view(2,1)
        res = input_test(a)
        loss = l2loss(gt,res)
        loss.backward()
        optimizer.step()
        print(loss)
    for param in input_test.parameters():
        print(param.data)
    print(input_test.fc1.parameters)

if __name__ == '__main__':
    main()

