# 自己处理好的数据集，已经划分好测试集，训练集，数据库集
import torch
import model
import numpy as np
from PIL import Image
import cv2
import config as conf
import math
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models


def Cal_Sim(label):
    size = len(label)
    Sim = label.mm(label.t()).float()
    V = torch.sum(label, dim=1).float()
    V1 = V.unsqueeze(0).t()
    S = V1.repeat([1, size])
    S = S + V - Sim
    Sim = Sim / S
    return Sim


class CocoDataset(Dataset):
    def __init__(self, img_list_path, label_path, tag_path, onestage_hashcode_path=None, transform=None):
        with open(img_list_path, 'r') as f:
            self.image_list = [line[:-1] for line in f]
        self.labels = np.load(label_path)
        self.tags = np.load(tag_path)

        self.transform = transform
        if onestage_hashcode_path is not None:
            self.one_stage_hash_code = np.load(onestage_hashcode_path)
        else:
            self.one_stage_hash_code = None

    def __getitem__(self, idx):
        img_name = self.image_list[idx]

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = torch.from_numpy(self.labels[idx]).float()
        tag = torch.from_numpy(self.tags[idx]).float()
        if self.transform:
            img = self.transform(img)
        # vgg-f时才用到
        img = img * 255.0

        if self.one_stage_hash_code is not None:
            one_stage_hash_code = self.one_stage_hash_code[idx]
            one_stage_hash_code = torch.from_numpy(one_stage_hash_code).float()
            one_stage_hash_code = one_stage_hash_code * 2 - 1
        else:
            one_stage_hash_code = -1
        return img, tag, label, one_stage_hash_code

    def __len__(self):
        return len(self.labels)


def data_loader(path):
    transformations_t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformations_q = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # mean and std of ImageNet
    ])

    # transformations = None
    # transformations = transforms.ToTensor()

    # 路径（包括测试集，训练集，数据库集）
    img_q_path = 'cross_coco_demo/query/image_list.txt'
    label_q_path = 'cross_coco_demo/query/label.npy'
    tag_q_path = 'cross_coco_demo/query/tag.npy'

    img_t_path = 'cross_coco_demo/train/image_list.txt'
    label_t_path = 'cross_coco_demo/train/label.npy'
    tag_t_path = 'cross_coco_demo/train/tag.npy'

    img_d_path = 'cross_coco_demo/database/image_list.txt'
    label_d_path = 'cross_coco_demo/database/label.npy'
    tag_d_path = 'cross_coco_demo/database/tag.npy'
    test_loader = DataLoader(CocoDataset(img_q_path, label_q_path, tag_q_path,
                                         onestage_hashcode_path=None, transform=transformations_t),
                             batch_size=conf.batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(CocoDataset(img_t_path, label_t_path, tag_t_path,
                                          onestage_hashcode_path=path, transform=transformations_q),
                              batch_size=conf.batch_size, shuffle=True, num_workers=4)
    database_loader = DataLoader(CocoDataset(img_d_path, label_d_path, tag_d_path,
                                             onestage_hashcode_path=None, transform=transformations_q),
                                 batch_size=conf.batch_size, shuffle=False, num_workers=4)
    return test_loader, train_loader, database_loader


def create_model(model_path, use_gpu):
    img_net = model.ImageNet(conf.bit, model_path, 0.5, conf.num_class)
    txt_net = model.TextNet(conf.y_dim, conf.bit, conf.num_class)
    if use_gpu:
        img_net = img_net.cuda()
        txt_net = txt_net.cuda()
    return img_net, txt_net


def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S


def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]).cuda())
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]))
    return lt


def train(test_loader, train_loader, database_loader, model_path, use_gpu):
    img_net, txt_net = create_model(model_path, use_gpu)

    lr = conf.lr

    my_params = list(img_net.fc8.parameters()) + list(img_net.classifier.parameters()) + list(
        img_net.hashlayer.parameters())
    base_params = list(map(id, my_params))
    cnnf_params = filter(lambda p: id(p) not in base_params, img_net.parameters())

    optimizer1 = optim.Adam([{'params': cnnf_params, 'lr': lr / 10},
                             {'params': img_net.classifier.parameters(), 'lr': lr},
                             {'params': img_net.hashlayer.parameters(), 'lr': lr}])

    optimizer2 = optim.Adam(txt_net.parameters(), lr=lr)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=1, gamma=0.9)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.9)

    max_mapi2t = max_mapt2i = 0.
    criterion = lambda x, y: (((x - y) ** 2).sum(1).sqrt()).mean()
    W = math.log(math.e + math.pow(conf.bit/conf.num_class, 2))-1

    for epoch in range(50):
        img_net.train()
        txt_net.train()
        scheduler1.step()
        scheduler2.step()

        hash_loss_i, classifier_loss_i, hash_loss_t, classifier_loss_t = 0.0, 0.0, 0.0, 0.0
        for img, tag, label, B_temp in train_loader:
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            tag = tag.float()
            if use_gpu:
                img, tag, label, B_temp = img.cuda(), tag.cuda(), label.cuda(), B_temp.cuda()

            # 训练图片网络
            cur_i, cur_i_l = img_net(img)

            S = CalcSim(label, label)
            S = S.cuda()

            label = label.cpu()
            S1 = Cal_Sim(label)
            S1 = S1.cuda()
            label = label.cuda()
            hashcode = torch.sign(cur_i.detach())
            S2 = (1 + hashcode.mm(hashcode.t())/conf.bit)/2

            # WCE损失
            theta_x = cur_i.mm(cur_i.t()) / 2
            log_x = Logtrick(theta_x, use_gpu)

            log_x1 = Logtrick(1-theta_x, use_gpu)
            # 参数需要重新调整
            gamma1 = 0.1
            gamma2 = 0.5
            logloss = (gamma1 * S1 * S * log_x + gamma2 * S2 * (1-S) * log_x1).sum() / (len(label) * len(label))
            # logloss = (0.1*S1 * S * theta_x - S1 * S * log_x - 0.5*S2*(1-S) * log_x).sum() / (len(label) * len(label))

            regterm = criterion(cur_i, B_temp)
            label_pred_term = criterion(cur_i_l, label)
            loss_i = -0.5 * 10 * logloss + regterm + W * label_pred_term

            loss_i.backward()
            optimizer1.step()
            hash_loss_i += float(loss_i.item())
            # cur_i_temp = cur_i.detach()
            del cur_i, cur_i_l

            # 训练文本网络
            cur_t, cur_t_l = txt_net(tag)

            # WCE损失
            hashcode = torch.sign(cur_t.detach())
            S2 = (1 + hashcode.mm(hashcode.t()) / conf.bit)/2
            theta_y = cur_t.mm(cur_t.t()) / 2
            log_y = Logtrick(theta_y, use_gpu)

            log_y1 = Logtrick(1 - theta_y, use_gpu)
            # 参数需要重新调整
            gamma1 = 0.1
            gamma2 = 0.5
            logloss = (gamma1 * S1 * S * log_y + gamma2 * S2 * (1-S) * log_y1).sum() / (len(label) * len(label))
            # logloss = (0.1*S1 * S * theta_y - S1 * S * log_y - 0.5*S2 * (1 - S) * log_y).sum() / (len(label) * len(label))

            regterm = criterion(cur_t, B_temp)
            label_pred_term1 = criterion(cur_t_l, label)
            loss_t = -1 * 10 * logloss + regterm + W * label_pred_term1

            loss_t.backward()
            optimizer2.step()
            hash_loss_t += float(loss_t.item())
            del cur_t, cur_t_l

        img_net.eval()
        txt_net.eval()
        num = len(train_loader)
        print('...epoch: %3d, img_loss: %3.3f, txt_loss: %3.3f' % (epoch + 1, hash_loss_i / num, hash_loss_t / num))
        mapi2t, mapt2i = valid(img_net, txt_net, test_loader, database_loader, use_gpu)
        map_sum = mapi2t + mapt2i
        print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, SUM: %3.4f' % (epoch + 1, mapi2t, mapt2i, map_sum))
        if mapi2t + mapt2i >= max_mapi2t + max_mapt2i:
            max_mapi2t = mapi2t
            max_mapt2i = mapt2i

            img_net.module_name = 'img_net'
            txt_net.module_name = 'txt_net'
            img_net.save(img_net.module_name + '.pth')
            txt_net.save(txt_net.module_name + '.pth')

    print('...training procedure finish')
    print('max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))


def generate_hashcode(dataloader, img_net, txt_net, use_gpu):
    bs, tags, clses = [], [], []
    with torch.no_grad():
        for img, tag, label, _ in dataloader:

            clses.append(label)
            if use_gpu:
                img, tag = img.cuda(), tag.cuda()
            # img = torch.tanh(img_net(img)[0].cpu().float())
            img = img_net(img)[0].cpu().float()
            img_b = torch.sign(img)
            if len(img.shape) == 1:
                img_b = img_b.reshape(1, img.shape[0])
            bs.append(img_b)

            # tag = torch.tanh(txt_net(tag)[0].cpu().float())
            tag = txt_net(tag)[0].cpu().float()
            tag_b = torch.sign(tag).cpu().float()
            tags.append(tag_b)
    return torch.cat(bs), torch.cat(tags), torch.cat(clses)


def valid(img_net, txt_net, test_loader, database_loader, use_gpu):
    qBX, qBY, query_L = generate_hashcode(test_loader, img_net, txt_net, use_gpu)
    rBX, rBY, retrieval_L = generate_hashcode(database_loader, img_net, txt_net, use_gpu)
    mapi2t = Calculate_mAP(query_L, qBX, retrieval_L, rBY, 500)
    mapt2i = Calculate_mAP(query_L, qBY, retrieval_L, rBX, 500)
    return mapi2t, mapt2i


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


# 计算mAP指标
def Calculate_mAP(tst_label, tst_binary, db_label, db_binary, top_k=-1):
    if top_k == -1:
        top_k = db_binary.size(0)
    mAP = 0.
    # 检索项的个数
    num_query = tst_binary.size(0)
    NS = (torch.arange(top_k) + 1).float()
    for idx in range(num_query):
        query_label = tst_label[idx].unsqueeze(0)
        query_binary = tst_binary[idx]
        result = query_label.mm(db_label.t()).squeeze(0) > 0
        # 数据库中图片按照与查询项的相似性排序
        hamm = calc_hammingDist(query_binary, db_binary)
        _, index = hamm.sort()
        index.squeeze_()
        result = result[index[: top_k]].float()
        # 数据库中与查询项有相同标签的图片数
        tsum = torch.sum(result)
        if tsum == 0:
            continue
        accuracy = torch.cumsum(result, dim=0) / tsum
        mAP += float(torch.sum(result * accuracy / NS).item())
    return mAP/num_query


def generate_hash_code(dataloader, img_net, txt_net, use_gpu):
    bs, tags, clses = [], [], []
    for img, tag, label, _ in dataloader:
        clses.append(label)
        if use_gpu:
            img, tag = img.cuda(), tag.cuda()
        img = torch.sigmoid(img_net(img)[0])
        img_b = (img >= 0.5).cpu().float()
        img_b = img_b * 2 - 1
        bs.append(img_b)

        tag = torch.sigmoid(txt_net(tag)[0])
        tag_b = (tag >= 0.5).cpu().float()
        tag_b = tag_b * 2 - 1
        tags.append(tag_b)
    return torch.sign(torch.cat(bs)), torch.sign(torch.cat(tags)), torch.cat(clses)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('读取数据')
    train_hashcode_path = conf.one_stage_hashcode_path
    test_loader, train_loader, database_loader = data_loader(train_hashcode_path)
    use_gpu = torch.cuda.is_available()

    # 注：11服务器设置0才是2号gpu
    if use_gpu:
        torch.cuda.set_device(3)

    model_path = '/home/qjq/pretrain_model/imagenet-vgg-f.mat'
    train(test_loader, train_loader, database_loader, model_path, use_gpu)
