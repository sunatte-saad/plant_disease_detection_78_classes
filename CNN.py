import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            # conv2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # conv3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            # conv4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            # conv5
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out


idx_to_classes = {0: 'Anthracnose',
                1: 'Apple___Apple_scab',
                2: 'Apple___Black_rot',
                3: 'Apple___Cedar_apple_rust',
                4: 'Apple___healthy',
                5: 'Background_without_leaves',
                6: 'Banana___cordana',
                7: 'Banana___healthy',
                8: 'Banana___pestalotiopsis',
                9: 'Banana___sigatoka',
                10: 'Blueberry___healthy',
                11: 'Cherry___Powdery_mildew',
                12: 'Cherry___healthy',
                13: 'Citrus_leaf____Black_spot',
                14: 'Citrus_leaf____Canker',
                15: 'Citrus_leaf____Greening',
                16: 'Citrus_leaf____Healthy',
                17: 'Citrus_leaf____Melanose',
                18: 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                19: 'Corn___Common_rust',
                20: 'Corn___Northern_Leaf_Blight',
                21: 'Corn___healthy',
                22: 'Grape___Black_rot',
                23: 'Grape___Esca_(Black_Measles)',
                24: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                25: 'Grape___healthy',
                26: 'Orange___Haunglongbing_(Citrus_greening)',
                27: 'Peach___Bacterial_spot',
                28: 'Peach___healthy',
                29: 'Pepper,_bell___Bacterial_spot',
                30: 'Pepper,_bell___healthy',
                31: 'Potato___Early_blight',
                32: 'Potato___Late_blight',
                33: 'Potato___healthy',
                34: 'Raspberry___healthy',
                35: 'Soybean___healthy',
                36: 'Squash___Powdery_mildew',
                37: 'Strawberry___Leaf_scorch',
                38: 'Strawberry___healthy',
                39: 'Tomato___Bacterial_spot',
                40: 'Tomato___Early_blight',
                41: 'Tomato___Late_blight',
                42: 'Tomato___Leaf_Mold',
                43: 'Tomato___Septoria_leaf_spot',
                44: 'Tomato___Spider_mites Two-spotted_spider_mite',
                45: 'Tomato___Target_Spot',
                46: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                47: 'Tomato___Tomato_mosaic_virus',
                48: 'Tomato___healthy',
                49: 'Wheat___Brown_rust',
                50: 'Wheat___Healthy',
                51: 'Wheat___Yellow_rust',
                52: 'aloe_vera___healthy',
                53: 'aloe_vera___rot',
                54: 'aloe_vera___rust',
                55: 'date_palm____brown_spots',
                56: 'date_palm____healthy',
                57: 'date_palm____white_scale',
                58: 'fig___healthy',
                59: 'fig___infected',
                60: 'guava____Canker',
                61: 'guava____Dot',
                62: 'guava____Healthy',
                63: 'guava____Mummification',
                64: 'guava____Rust',
                65: 'lettuce____Bacterial',
                66: 'lettuce____fungal',
                67: 'lettuce____healthy',
                68: 'rice____bacterial_leaf_blight',
                69: 'rice____brown_spot',
                70: 'rice____healthy',
                71: 'rice____leaf_blast',
                72: 'rice____leaf_scald',
                73: 'rice____narrow_brown_spot',
                74: 'water_melon____Anthracnose',
                75: 'water_melon____Downy_Mildew',
                76: 'water_melon____Healthy',
                77: 'water_melon____Mosaic_Virus'}
                