import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        '''
        outputs: [b , c , h , w] , if outputs are multi scale ,get first element
        '''
        if type(outputs) == list:
            outputs = outputs[0]
        target_size = targets.size()[:-2]
        outputs = F.upsample(outputs, size=target_size, mode='bilinear')
        return self.loss(self.softmax(outputs), targets)


class Multi_Scale_CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(Multi_Scale_CrossEntropyLoss2d, self).__init__()
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        '''
        ouptus: a list of [b , c,  h , w]
        targets: [b , h  , w]
        '''
        total_loss = 0
        target_size = targets.size()[-2:]
        for output in outputs:
            output = F.upsample(output, size=target_size, mode='bilinear')
            total_loss += self.loss(self.softmax(output), targets)
        return total_loss


class Fscore_Loss(nn.Module):
    def __init__(self, bias=1):
        super(Fscore_Loss, self).__init__()
        self.bias = bias
        self.softmax = nn.LogSoftmax()

    def forward(self, outpus, target):
        '''
        ouptus: [b , c,  h , w] if outpus are multi scale, calculate multi loss
        targets: [b , h  , w]
        '''
        total_loss = 0
        target_size = targets.size()[:-2]
        if type(outputs) == list:
            for output in outputs:
                output = F.upsample(output, size=target_size, mode='bilinear')
                total_loss += self.floss(self.softmax(output), targets)
        else:
            output = F.upsample(outputs, size=target_size, mode='bilinear')
            total_loss += self.floss(self.softmax(output), targets)
        return total_loss

    def floss(self, output, target):
        '''
        output: [0-1] , [b ,c , h , w]
        target: [0 , 1] , [b , h ,w]
        '''
        target.unsqueeze_(1)
        one_hot = torch.FloatTensor(
            target.size(0), 2, target.size(2), target.size(3)).zero_().cuda()
        target = one_hot.scatter_(1, target.data, 1)  # [b , c, h , w]
        print('target.size()', target.size())
        TP = torch.sum(output * target)
        FP = torch.sum(output * (1 - target))
        FN = torch.sum((1 - output) * target)
        H = self.bias**2 * (TP + FN) + (TP + FP) + 1e-10
        return (1 + self.bias**2) / H