import torch

def sub_max(input):
    with torch.no_grad():
        temp = input.permute(1,0).bfloat16().float()
        input.data = temp.add(-temp.view(-1, temp.size(1)).max(dim=0)[0]).permute(1,0).data
    return input

def sub_max_rnn(input):
    with torch.no_grad():
        temp = input.permute(0,2,1)
        input.data = temp.add(-temp.reshape(-1,temp.size(2)).max(dim=0)[0]).permute(0,2,1).data
    return input