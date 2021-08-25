import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms

import os
import argparse

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x.view(-1, 32*7*7))
        return x

def train(gpu, args):
    rank = args.rank * args.gpus + gpu
    dist.init_process_group(
        backend='nccl', init_method = 'env://',
        world_size = args.world_size,
        rank = rank
    )
    print(f"process {rank} beginning training")
    torch.manual_seed(args.seed)

    net = Net()
    torch.cuda.set_device(gpu)
    net.cuda(gpu) # can I just do .to(...) here?

    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transforms, download=True)
    test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transforms, download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas = args.world_size, rank = rank
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas = args.world_size, rank = rank
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, sampler=test_sampler)

    for eid in range(args.nb_epochs):
        net.train()
        train_loss, test_loss = 0.0, 0.0
        for x, y in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = net(x)

            optim.zero_grad()
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optim.step()

            train_loss += loss.item()
        
        net.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logits = net(x)
                loss = F.cross_entropy(logits, y)

                test_loss += loss.item()

        if rank == 0:
            msg = (
                f"epoch {eid+1}/{args.nb_epochs} | "
                f"train loss: {train_loss / len(train_loader)} | "
                f"test loss: {test_loss / len(test_loader)}"
            )
            print(msg)

    print(f"process {rank} terminating")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int)
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('-r', '--rank', default=0, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--nb-epochs', default=100, type=int)
    parser.add_argument('--address', default='127.0.0.1', type=str)
    parser.add_argument('--port', default='12345', type=str)
    args = parser.parse_args()

    args.world_size = args.nodes * args.gpus
    os.environ['MASTER_ADDR'] = args.address
    os.environ['MASTER_PORT'] = args.port
    mp.spawn(train, nprocs=args.gpus, args=(args,))

