import torch.nn.functional as F
import torch

def train(loader, model, optimizer, epoch, cuda, log_interval, verbose=True):
    model.train()
    global_epoch_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.data.item()
        if verbose:
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset), 100.
                    * batch_idx / len(loader), loss.data.item()))
    if verbose:                             
        print('\nTrain loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            global_epoch_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
        
    return global_epoch_loss / len(loader.dataset)


def test(loader, model, cuda, is_valid=False, verbose=True):
    with torch.no_grad(): 
        model.eval()
        loss = 0
        correct = 0
        data_type = 'Valid' if is_valid else 'Test' 
        for data, target in loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss /= len(loader.dataset)
        acc = correct / len(loader.dataset)
        if verbose:
            print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                data_type, loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
        return loss, acc
    