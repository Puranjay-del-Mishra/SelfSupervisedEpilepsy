import torch
from torch import nn

input_size = 4096 # random number, upto change

class eeg_generator(nn.Module):
    def __init__(self):
        super(eeg_generator, self).__init__()
        self.conv1d_1 = nn.Conv1d(23, 16, kernel_size = 5, stride = 2,padding=2)
        # self.conv1d_2 = nn.Conv1d(16, 16, kernel_size = 5, stride = 2,padding=2)
        # self.conv1d_3 = nn.Conv1d(16,16,kernel_size=5, stride=2, padding=2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=4096,nhead=32)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)

        # self.decoder_layer = nn.TransformerDecoderLayer(d_model= 2048, nhead= 8)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers= 4)
        # self.conv1d_4 = nn.Conv1d(16,16,kernel_size=5, stride=2)
        # self.conv1d_5 = nn.Conv1d(16,16,kernel_size=5, stride=2)
        self.conv1d_6 = nn.Conv1d(16, 23, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(2046,256)
        self.linear2 = nn.Linear(256,8192)
        self.relu = nn.ReLU()
        self.norm = nn.functional.normalize

    def forward(self,inp):
        x = self.conv1d_1(inp)
        x = self.norm(x,dim=-1)
        # x = self.conv1d_2(x)
        # x = self.norm(x,dim=-1)
        # x = self.conv1d_3(x)
        # x = self.norm(x)
        x = self.encoder(x)
        # # x = self.transform(x)
        # tgt = x
        # memory = x
        # x = self.decoder(tgt, memory)
        # x = self.norm(x,dim=-1)
        # x = self.conv1d_4(x)
        # x = self.norm(x, dim=-1)
        # x = self.conv1d_5(x)
        x = self.norm(x, dim=-1)
        x = self.conv1d_6(x)
        x = self.norm(x, dim=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.norm(x,dim=-1)
        x = self.linear2(x)
        return x

class linear_eeg_generator(nn.Module):
    def __init__(self):
        super(linear_eeg_generator,self).__init__()
        self.lin1 = nn.Linear(12288,256)
        self.lin2 = nn.Linear(256,256)
        self.lin3 = nn.Linear(256,128)
        self.lin4 = nn.Linear(128,256)
        self.lin5 = nn.Linear(256,256)
        self.lin6 = nn.Linear(256,12288)
        self.relu = nn.ReLU()
        self.norm = nn.functional.normalize
    def forward(self,x):
        x = self.norm(x,dim=-1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.norm(x,dim=-1)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.norm(x,dim=-1)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.norm(x, dim=-1)
        x = self.lin4(x)
        x = self.relu(x)
        x = self.norm(x, dim=-1)
        x = self.lin5(x)
        x = self.relu(x)
        x = self.norm(x, dim=-1)
        x = self.lin6(x)
        x = self.relu(x)
        return x

class conv_linear_generator(nn.Module):
    def __init__(self):
        super(conv_linear_generator, self).__init__()
        self.conv1d_1 = nn.Conv1d(23, 16, kernel_size = 5, stride = 2,padding=1)
        self.conv1d_2 = nn.Conv1d(16, 16, kernel_size = 5, stride = 2,padding=2)
        self.conv1d_3 = nn.Conv1d(16,16,kernel_size=5, stride=2, padding=2)
        self.conv1d_4 = nn.Conv1d(16, 23, kernel_size=5, stride=2)
        self.lin1 = nn.Linear(1022,512)
        self.lin2 = nn.Linear(512,input_size)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

class frozen_encoder(nn.Module):
    def __init__(self,generator):
        super(frozen_encoder, self).__init__()
        self.conv1d_1 = generator.conv1d_1
        self.conv1d_2 = generator.conv1d_2
        self.encoder = generator.encoder
    def forward(self, inp):
        x = self.conv1d_1(inp)
        x = self.conv1d_2(x)
        x = self.encoder(x)
        return x
