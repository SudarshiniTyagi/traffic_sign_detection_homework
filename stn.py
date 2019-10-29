import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 200, kernel_size=7, padding=2)
        self.conv2 = nn.Conv2d(200, 250, kernel_size=4, padding=2)
        self.conv3 = nn.Conv2d(250, 350, kernel_size=4, padding=2)
        self.conv4 = nn.Conv2d(350, 350, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12600, 400)
        self.fc2 = nn.Linear(400, nclasses)
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.localization1 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(3, 250, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(250, 250, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(9000, 250),
            nn.ReLU(),
            nn.Linear(250, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.localization2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(200, 150, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(150, 200, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(800, 300),
            nn.ReLU(),
            nn.Linear(300, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


        self.localization3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(250, 150, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(150, 200, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc3 = nn.Sequential(
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc3[2].weight.data.zero_()
        self.fc_loc3[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        
        
        
    def LocalContrastNorm(self, image, radius=9):
        """
        image: torch.Tensor , .shape => (1,channels,height,width)

        radius: Gaussian filter size (int), odd
        """

        if radius % 2 == 0:
            radius += 1

        def get_gaussian_filter(kernel_shape):
            x = np.zeros(kernel_shape, dtype='float64')

            def gauss(x, y, sigma=2.0):
                Z = 2 * np.pi * sigma ** 2
                return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

            mid = np.floor(kernel_shape[-1] / 2.)
            for kernel_idx in range(0, kernel_shape[1]):
                for i in range(0, kernel_shape[2]):
                    for j in range(0, kernel_shape[3]):
                        x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

            return x / np.sum(x)

        n, c, h, w = image.shape[0], image.shape[1], image.shape[2], image.shape[3]

        gaussian_filter = torch.Tensor(get_gaussian_filter((1, c, radius, radius)))
        image = image.to("cpu")
        filtered_out = F.conv2d(image, gaussian_filter, padding=radius - 1)
        mid = int(np.floor(gaussian_filter.shape[2] / 2.))
        ### Subtractive Normalization
        centered_image = image - filtered_out[:, :, mid:-mid, mid:-mid]

        ## Variance Calc
        sum_sqr_image = F.conv2d(centered_image.pow(2), gaussian_filter, padding=radius - 1)
        s_deviation = sum_sqr_image[:, :, mid:-mid, mid:-mid].sqrt()
        per_img_mean = s_deviation.mean()

        ## Divisive Normalization
        divisor = np.maximum(per_img_mean.detach().numpy(), s_deviation.detach().numpy())
        divisor = np.maximum(divisor, 1e-4)
        new_image = centered_image / torch.Tensor(divisor)
        new_image = new_image.to("cuda:0")
        return new_image
    

    def stn1(self, x):
        xs = self.localization1(x)
        xs = xs.view(-1, 9000)
        theta = self.fc_loc1(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def stn2(self, x):
        xs = self.localization2(x)
        xs = xs.view(-1, 800)
        theta = self.fc_loc2(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def stn3(self, x):
        xs = self.localization3(x)
        xs = xs.view(-1, 200)
        theta = self.fc_loc3(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


    def forward(self, x):

        x = self.stn1(x)

        #64x3x48x48
        # print("1. ",x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.LocalContrastNorm(x)

        x = self.stn2(x)


        #64x200x23x23
        # print("2. ",x.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.LocalContrastNorm(x)

        x = self.stn3(x)

        # 64x250x12x12
        # print("3. ", x.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = self.LocalContrastNorm(x)

        # x = F.relu(self.conv4(x))


        #64x350x6x6
        # print("4. ",x.shape)
        x = x.view(-1, 12600)

        #64x12600
        # print("5. ", x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)


        #64x400
        # print("6. ",x.shape)
        x = self.fc2(x)

        #64x43
        # print("7. ",x.shape)
        # return F.log_softmax(x)
        return x