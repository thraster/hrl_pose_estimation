import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kinematics_lib import KinematicsLib
import scipy.stats as ss
import torchvision
import resnet

class CNN(nn.Module):
    def __init__(self, mat_size, out_size, hidden_dim, kernel_size, loss_vector_type):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            mat_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            out_size (int): Number of classes to score
        '''

        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #print mat_size
        self.loss_vector_type = loss_vector_type

        hidden_dim1= 32
        hidden_dim2 = 48
        hidden_dim3 = 96
        hidden_dim4 = 96

        self.count = 0

        # ConvNet的整个backbone部分
        self.CNN_pack1 = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
        )

        # ConvNet的backbone的开头部分：7x7conv
        self.CNN_pack2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.ReLU(inplace = True),
            #nn.Dropout(p = 0.1, inplace=False),
            #nn.MaxPool2d(3, stride=2),
            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1, inplace=False),
            #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1, inplace=False),
        
        )


        # ConvNet的backbone的vgg子核：3x3conv
        self.CNN_pack3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),

        )

        # vgg全连接层部分的实现
        self.VGG_fc1 = nn.Sequential(
            nn.Linear(12288, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_size),
        )


        print ('x')
        # 一层全连接层的实现
        self.CNN_fc1 = nn.Sequential(
            '''
            一层全连接层
            '''
            nn.Linear(22400, out_size),

        )

        # self.resnet18 = torchvision.models.resnet18(pretrained=False, num_classes=out_size)
        print( 'Out size:', out_size)
        # self.resnet = resnet.resnet18(pretrained=True)
        # self.resnet = resnet.resnet34(pretrained=True)
        # self.resnet = resnet.resnet18(pretrained=False, num_classes=out_size)
        # self.resnet = resnet.resnet34(pretrained=False, num_classes=out_size)
        # self.resnet = resnet.resnet50(pretrained=False, num_classes=out_size)
        # self.resnet = resnet.resnet101(pretrained=False, num_classes=out_size)


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward_direct(self, images, targets):

        '''
        1. Direct ConvNet的实现

        Take a batch of images and run them through the CNN to
        produce a scores for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, out_size) specifying the scores
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #print images.size(), 'CNN input size'

        # 1. images先通过vgg卷积层
        scores = self.CNN_pack1(images)
        scores_size = scores.size()
        #print scores_size, 'scores conv1'



        #scores = self.CNN_pack4(scores)
        #scores_size = scores.size()
        #print scores_size, 'scores conv4'


        # This combines the height, width, and filters into a single dimension
        scores = scores.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3])

        #print scores.size(), 'scores fc1'


        # 2. vgg卷积层出来之后进入一个全连接层fc1
        scores = self.CNN_fc1(scores)


        scores[:, 0] = torch.add(scores[:, 0], 0.6)
        scores[:, 1] = torch.add(scores[:, 1], 1.3)
        scores[:, 2] = torch.add(scores[:, 2], 0.1)
        scores[:, 3] = torch.add(scores[:, 3], 0.6)
        scores[:, 4] = torch.add(scores[:, 4], 1.3)
        scores[:, 5] = torch.add(scores[:, 5], 0.1)
        scores[:, 6] = torch.add(scores[:, 6], 0.6)
        scores[:, 7] = torch.add(scores[:, 7], 1.3)
        scores[:, 8] = torch.add(scores[:, 8], 0.1)
        scores[:, 9] = torch.add(scores[:, 9], 0.6)
        scores[:, 10] = torch.add(scores[:, 10], 1.3)
        scores[:, 11] = torch.add(scores[:, 11], 0.1)
        scores[:, 12] = torch.add(scores[:, 12], 0.6)
        scores[:, 13] = torch.add(scores[:, 13], 1.3)
        scores[:, 14] = torch.add(scores[:, 14], 0.1)
        scores[:, 15] = torch.add(scores[:, 15], 0.6)
        scores[:, 16] = torch.add(scores[:, 16], 1.3)
        scores[:, 17] = torch.add(scores[:, 17], 0.1)
        scores[:, 18] = torch.add(scores[:, 18], 0.6)
        scores[:, 19] = torch.add(scores[:, 19], 1.3)
        scores[:, 20] = torch.add(scores[:, 20], 0.1)
        scores[:, 21] = torch.add(scores[:, 21], 0.6)
        scores[:, 22] = torch.add(scores[:, 22], 1.3)
        scores[:, 23] = torch.add(scores[:, 23], 0.1)
        scores[:, 24] = torch.add(scores[:, 24], 0.6)
        scores[:, 25] = torch.add(scores[:, 25], 1.3)
        scores[:, 26] = torch.add(scores[:, 26], 0.1)
        scores[:, 27] = torch.add(scores[:, 27], 0.6)
        scores[:, 28] = torch.add(scores[:, 28], 1.3)
        scores[:, 29] = torch.add(scores[:, 29], 0.1)

        targets_est = scores.clone().data*1000.

        #print scores.size(), 'scores fc2'
        #here we want to compute our score as the Euclidean distance between the estimated x,y,z points and the target.

        scores = targets/1000. - scores
        scores = scores.pow(2)
        scores[:, 0] = scores[:, 0] + scores[:, 1] + scores[:, 2]
        scores[:,1] = scores[:,3]+scores[:,4]+scores[:,5]
        scores[:,2] = scores[:,6]+scores[:,7]+scores[:,8]
        scores[:,3] = scores[:,9]+scores[:,10]+scores[:,11]
        scores[:,4] = scores[:,12]+scores[:,13]+scores[:,14]
        scores[:,5] = scores[:,15]+scores[:,16]+scores[:,17]
        scores[:,6] = scores[:,18]+scores[:,19]+scores[:,20]
        scores[:,7] = scores[:,21]+scores[:,22]+scores[:,23]
        scores[:,8] = scores[:,24]+scores[:,25]+scores[:,26]
        scores[:,9] = scores[:,27]+scores[:,28]+scores[:,29]
        scores = scores[:, 0:10]
        scores = scores.sqrt()


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores, targets_est


    def forward_direct_vgg(self, vgg_image_features, targets):
        #print vgg_image_features.shape
        vgg_sz = vgg_image_features.size()
        vgg_image_features = vgg_image_features.view(vgg_sz[0],vgg_sz[1]*vgg_sz[2]*vgg_sz[3])
        #print vgg_image_features.shape


        scores = self.VGG_fc1(vgg_image_features)


        # 这段代码对 scores 进行了逐元素的增加操作。
        # 在这个上下文中，scores 是神经网络输出的一组得分或预测值，每一列对应不同的预测。
        # 这个操作的目的可能是为了调整网络的预测结果。

        # 具体地说，对于每一列（每个预测），都增加了一个固定的值，这些值是手动指定的。
        # 例如，对于第一列，增加了0.6；对于第二列，增加了1.3，以此类推。
        # 这样的操作可以在一定程度上调整神经网络的预测结果，以使其更接近预期的输出。
        scores[:, 0] = torch.add(scores[:, 0], 0.6)
        scores[:, 1] = torch.add(scores[:, 1], 1.3)
        scores[:, 2] = torch.add(scores[:, 2], 0.1)
        scores[:, 3] = torch.add(scores[:, 3], 0.6)
        scores[:, 4] = torch.add(scores[:, 4], 1.3)
        scores[:, 5] = torch.add(scores[:, 5], 0.1)
        scores[:, 6] = torch.add(scores[:, 6], 0.6)
        scores[:, 7] = torch.add(scores[:, 7], 1.3)
        scores[:, 8] = torch.add(scores[:, 8], 0.1)
        scores[:, 9] = torch.add(scores[:, 9], 0.6)
        scores[:, 10] = torch.add(scores[:, 10], 1.3)
        scores[:, 11] = torch.add(scores[:, 11], 0.1)
        scores[:, 12] = torch.add(scores[:, 12], 0.6)
        scores[:, 13] = torch.add(scores[:, 13], 1.3)
        scores[:, 14] = torch.add(scores[:, 14], 0.1)
        scores[:, 15] = torch.add(scores[:, 15], 0.6)
        scores[:, 16] = torch.add(scores[:, 16], 1.3)
        scores[:, 17] = torch.add(scores[:, 17], 0.1)
        scores[:, 18] = torch.add(scores[:, 18], 0.6)
        scores[:, 19] = torch.add(scores[:, 19], 1.3)
        scores[:, 20] = torch.add(scores[:, 20], 0.1)
        scores[:, 21] = torch.add(scores[:, 21], 0.6)
        scores[:, 22] = torch.add(scores[:, 22], 1.3)
        scores[:, 23] = torch.add(scores[:, 23], 0.1)
        scores[:, 24] = torch.add(scores[:, 24], 0.6)
        scores[:, 25] = torch.add(scores[:, 25], 1.3)
        scores[:, 26] = torch.add(scores[:, 26], 0.1)
        scores[:, 27] = torch.add(scores[:, 27], 0.6)
        scores[:, 28] = torch.add(scores[:, 28], 1.3)
        scores[:, 29] = torch.add(scores[:, 29], 0.1)

        #print scores[0, :]

        # 将原始 scores 张量的克隆副本的数据部分放大了1000倍，然后将放大后的数据保存在 targets_est 变量中。
        # 可能的原因是为了使得 targets_est 与模型的输出在同一个数量级上，以便于后续计算和比较。
        targets_est = scores.clone().data*1000.

        #print scores.size(), 'scores fc2'

        #here we want to compute our score as the Euclidean distance between the estimated x,y,z points and the target.
        scores = targets/1000. - scores
        scores = scores.pow(2)
        scores[:, 0] = scores[:, 0] + scores[:, 1] + scores[:, 2]
        scores[:,1] = scores[:,3]+scores[:,4]+scores[:,5]
        scores[:,2] = scores[:,6]+scores[:,7]+scores[:,8]
        scores[:,3] = scores[:,9]+scores[:,10]+scores[:,11]
        scores[:,4] = scores[:,12]+scores[:,13]+scores[:,14]
        scores[:,5] = scores[:,15]+scores[:,16]+scores[:,17]
        scores[:,6] = scores[:,18]+scores[:,19]+scores[:,20]
        scores[:,7] = scores[:,21]+scores[:,22]+scores[:,23]
        scores[:,8] = scores[:,24]+scores[:,25]+scores[:,26]
        scores[:,9] = scores[:,27]+scores[:,28]+scores[:,29]
        scores = scores[:, 0:10]
        scores = scores.sqrt()


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores, targets_est

    def forward_confidence(self, images, targets_proj):
        scores_cnn = self.CNN_pack2(images)
        scores_size = scores_cnn.size()
        print scores_size, 'scores conv1'

        scores_cnn = self.CNN_pack3(scores_cnn)
        scores_size = scores_cnn.size()
        print scores_size, 'scores conv1'


        print images.size()
        print targets_proj.size()


    def forward_kinematic_jacobian(self, images, targets=None, kincons=None, forward_only = False, subject = None, loss_vector_type = None):
        scores = None
        targets_est = None
        lengths_est = None


        # 1. img通过ConvNet的backbone, 得到特征图scores_cnn
        scores_cnn = self.CNN_pack1(images)
        scores_size = scores_cnn.size()
        # print scores_size, 'scores conv1'

        # ''' # NOTE: Uncomment
        # This combines the height, width, and filters into a single dimension

        # 计算输出的特征图的长、宽、通道数，合并为一个一维向量
        # images.size(0)- batchsize
        # scores_size[1] *scores_size[2]*scores_size[3] = height*width*channel
        scores_cnn = scores_cnn.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3] )
        #print 'size for fc layer:', scores_cnn.size()

        # 2. 把特征图展开为一维向量后通过全连接层fc1, 得到scores
        scores = self.CNN_fc1(scores_cnn)
        # ''' # NOTE: Uncomment

        # print images.size()
        # scores = self.resnet(images) # NOTE: Uncomment

        #kincons_est = Variable(torch.Tensor(np.copy(scores.data.numpy())))

        #torso_scores = scores[:, 0:3]


        #angles_scores = scores[:, 11:19]

        if kincons is not None:
            kincons = kincons / 100

        print(loss_vector_type, 'LOSS VECT')

        # 3. 根据损失函数的类型来输入kinematics model, 返回结果
        if loss_vector_type == 'anglesCL' or loss_vector_type == 'anglesVL':
            scores, angles_est, pseudotargets_est = KinematicsLib().forward_kinematics_pytorch(images, scores, loss_vector_type, kincons, forward_only = forward_only, subject = subject, count = self.count)
        elif loss_vector_type == 'anglesSTVL':
            scores, angles_est, pseudotargets_est = KinematicsLib().forward_kinematics_lengthsv_pytorch(images, scores, loss_vector_type, kincons, forward_only = forward_only, subject = subject)


        #print scores.size(), ''

        targets_est = scores[:, 17:47].data*1000. #after it comes out of the forward kinematics
        targets_est[:, 0:3] = scores[:, 20:23].data*1000. #after it comes out of the forward kinematics
        targets_est[:, 3:6] = scores[:, 17:20].data*1000. #after it comes out of the forward kinematics
        lengths_est = scores[:, 0:17].data

        #tweak this to change the lengths vector
        scores[:, 0:17] = torch.mul(scores[:, 0:17], 1)

        if forward_only == False:
            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (10, 30, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

            #print scores.size()
            #print targets.size()

            scores[:, 27:30] = targets[:, 3:6]/1000 - scores[:, 27:30]
            scores[:, 30:33] = targets[:, 0:3]/1000 - scores[:, 30:33]
            scores[:, 33:57] = targets[:, 6:30]/1000 - scores[:, 33:57]
            scores[:, 57:87] = ((scores[:, 27:57])*1.).pow(2)
            self.count += 1
            if self.count < 300:
                scores[:, 0] = (scores[:, 57] + scores[:, 58] + scores[:, 59]).sqrt()*2# consider weighting the torso by a >1 factor because it's very important to root the other joints #bad idea, increases error
            elif self.count < 1000:
                scores[:, 0] = (scores[:, 57] + scores[:, 58] + scores[:, 59]).sqrt()*2# consider weighting the torso by a >1 factor because it's very important to root the other joints #bad idea, increases error
            else:
                scores[:, 0] = (scores[:, 57] + scores[:, 58] + scores[:, 59]).sqrt()*2
            scores[:, 1] = (scores[:, 60] + scores[:, 61] + scores[:, 62]).sqrt()
            scores[:, 2] = (scores[:, 63] + scores[:, 64] + scores[:, 65]).sqrt()
            scores[:, 3] = (scores[:, 66] + scores[:, 67] + scores[:, 68]).sqrt()
            scores[:, 6] = (scores[:, 75] + scores[:, 76] + scores[:, 77]).sqrt()
            scores[:, 7] = (scores[:, 78] + scores[:, 79] + scores[:, 80]).sqrt()
            #if self.count < 1500:
            #    scores[:, 4] = (scores[:, 69] + scores[:, 70] + scores[:, 71]).sqrt()*0.5
            #    scores[:, 5] = (scores[:, 72] + scores[:, 73] + scores[:, 74]).sqrt()*0.5
            #    scores[:, 8] = (scores[:, 81] + scores[:, 82] + scores[:, 83]).sqrt()*0.5
            #    scores[:, 9] = (scores[:, 84] + scores[:, 85] + scores[:, 86]).sqrt()*0.5
            #else:
            scores[:, 4] = (scores[:, 69] + scores[:, 70] + scores[:, 71]).sqrt()
            scores[:, 5] = (scores[:, 72] + scores[:, 73] + scores[:, 74]).sqrt()
            scores[:, 8] = (scores[:, 81] + scores[:, 82] + scores[:, 83]).sqrt()
            scores[:, 9] = (scores[:, 84] + scores[:, 85] + scores[:, 86]).sqrt()

            print self.count


            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -60, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return  scores, targets_est, angles_est, lengths_est, pseudotargets_est, #, lengths_scores
