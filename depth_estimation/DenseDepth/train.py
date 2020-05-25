import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import NYUDataset
from model import DenseDepth
from loss import DenseDepthLoss


cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(cuda)

bs = 4
grad_acc_step = 2

train_ds = NYUDataset('/mnt/hdd/dataset/NYU_Depth_V2/data/nyu2_train/', True, True)
test_ds = NYUDataset('/mnt/hdd/dataset/NYU_Depth_V2/data/nyu2_test/', False, True)
train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=1)

model = DenseDepth()
model.to(cuda)
criterion = DenseDepthLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(60):
    running_loss = 0.

    model.train()
    start_time = time.time()
    for i, sample in enumerate(train_dataloader):
        optimizer.zero_grad()

        image, depth = sample['image'].to(cuda), sample['depth'].to(cuda)

        pred = model(image)

        loss = criterion(pred, depth)
        running_loss += loss.item()
        loss = loss / grad_acc_step
        loss.backward()

        if i % grad_acc_step == grad_acc_step-1:
            optimizer.step()

            if i % 2000 == 1999:
                print('[{:2}, {:5}] loss: {:4f} / time : {:10}'.format(epoch+1,
                                                                       i+1,
                                                                       running_loss/2000.,
                                                                       int(time.time()-start_time)))
                start_time = time.time()
                running_loss = 0

    with torch.no_grad():
        test_loss = 0.
        for sample in test_dataloader:
            image, depth = sample['image'].to(cuda), sample['depth'].to(cuda)
            pred = model(image)
            loss = criterion(pred, depth)
            test_loss += loss.item()
        print('[---- {} epoch ----] loss : {}'.format(epoch, test_loss / (len(test_ds)//bs)))
        torch.save(model.state_dict(), 'model_saved/epoch_{}'.format(epoch))

