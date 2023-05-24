from models import eeg_generator, frozen_encoder, linear_eeg_generator, conv_linear_generator
from data_preprocessing import normal_save_loc, all_save_loc
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.cluster import MiniBatchKMeans
import pickle
import os
from torch import nn

checkpoint_loc = "C:/Users/puran/PycharmProjects/SelfSupervised_Seizure/checkpoint"


def main():
    def calc_acc(out, label):
        cnt = 0
        for i1, i2 in zip(out, label):
            if i1 == i2:
                cnt += 1
        return (cnt / len(out)) * 100

    class normal_eeg_dataset(Dataset):
        def __init__(self, data_loc):
            super().__init__()
            self.data_path_list = os.listdir(data_loc)
            self.data_loc = data_loc

        def __len__(self):
            return len(self.data_path_list)

        def __getitem__(self, idx):
            pickle_file = self.data_path_list[idx]
            # file_loc = os.path.join(self.data_loc,pickle_file)
            f = open(self.data_loc+"/"+pickle_file, "rb")
            temp_dict = pickle.load(f)
            data, masked_data = temp_dict['array'], temp_dict['masked']
            data = torch.tensor(data).float()
            masked_data = torch.tensor(masked_data).float()

            return data, masked_data

    class all_eeg_dataset(Dataset):
        def __init__(self, data_loc):
            super().__init__()
            self.data_path_list = os.listdir(data_loc)
            self.data_loc = data_loc

        def __len__(self):
            return len(self.data_path_list)

        def __getitem__(self, idx):
            pickle_file = self.data_path_list[idx]
            # file_loc = os.path.join(self.data_loc, pickle_file)
            temp_dict = pickle.load(open(pickle_file, "rb"))
            data = temp_dict['input']
            label = temp_dict['label']
            data = torch.tensor(data).float()
            label = torch.tensor(label).float()
            return data, label

    def CustomCosh(val,denominator):
        return torch.div((torch.exp(val)+torch.exp(-1*val)),(2+denominator))

    class LogCoshError(nn.Module):
        def __init__(self):
            super(LogCoshError, self).__init__()
        def forward(self,out,tar):
            dif = out-tar
            err = torch.cosh(dif)
            err = torch.log(err)
            # print(err)
            err = torch.sum(err)
            print(err)
            return err

    normal_eeg_data = normal_eeg_dataset(normal_save_loc)
    all_eeg_data = all_eeg_dataset(normal_save_loc)
    generator_data_loader = DataLoader(normal_eeg_data, batch_size=8, shuffle=True)
    all_data_loader = DataLoader(all_eeg_data, batch_size=8, shuffle=True)

    eeg_generator_model = eeg_generator()
    optim = torch.optim.RMSprop(eeg_generator_model.parameters(), lr=0.0001)
    loss = nn.HuberLoss(delta=3)
    epochs = 40
    cur_epoch = 0
    cnt = 0

    generator_loss_log = []
    clustering_loss_log = []
    resume_generator_from_checkpoint = False

    def load_model(checkpoint_loc,start):
        checkpoint = torch.load(checkpoint_loc)
        eeg_generator_model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        cur_epoch = checkpoint['epoch']
        generator_loss_log = checkpoint['generator_loss_log']
        if start:
            print(f'Resuming the training from the {cur_epoch}th epoch')
        else:
            print(f'Loaded the best available checkpoint for the {cur_epoch}th epoch')

    if resume_generator_from_checkpoint == True:
        load_model(checkpoint_loc,True)
    else:
        print('Training the model from scratch...')


    eeg_generator_model.train()
    best_loss = 10000
    for epoch in range(cur_epoch, epochs):
        for inp, mask_inp in generator_data_loader:
            optim.zero_grad()
            cnt += 1
            optim.zero_grad()
            out = eeg_generator_model(inp.float())

            l = loss(out, mask_inp)

            generator_loss_log.append(l.item())
            l.backward()
            optim.step()

            print('\n')
            print('Generator loss at the current mini epoch is =', l.item())

            if l.item()*0.85<best_loss:
                print('\n')
                checkpoint = {'model_state_dict': eeg_generator_model.state_dict(),
                              'optim_state_dict': optim.state_dict(),
                              'epoch': epoch, 'generator_loss_log': generator_loss_log}
                torch.save(checkpoint, checkpoint_loc)
                print('The current training step was close to optimal and was saved!')
                best_loss = l.item()
            else:
                load_model(checkpoint_loc, False)

    frozen_encoder_model = frozen_encoder(eeg_generator_model)

    for params in frozen_encoder_model.params():
        params.requires_grad = False

    mini_kmeans = MiniBatchKMeans()

    epochs = 40
    cur_epoch = 0
    cnt = 0
    file_name = 'mini_kmeans.pickle'
    acc_list = []

    resume_mini_kmeans_from_checkpoint = False
    training_state_loc = 'C:/Users/puran/PycharmProjects/Puranjay_attempt1/kmeans_epoch_state'

    if resume_mini_kmeans_from_checkpoint == True:
        mini_kmeans = pickle.load(open(file_name, "rb"))
        cur_state = torch.load(training_state_loc)
        cur_epoch = cur_state['epoch']
        acc_list = cur_state['acc_list']
        print(f'Resuming the training of kmeans model from {cur_epoch}th epoch')

    else:
        print('Training the Mini Batch Kmeans model from scratch!')

    ### we can use the existing labels for obtaining the accuracy and F1 score for the model later.

    for epoch in range(cur_epoch, epochs):
        for inp, label in all_data_loader:
            cnt += 1
            if cnt % 4 == 0:
                out = mini_kmeans.predict(inp)
                ##out is a ndarray
                acc = calc_acc(out, label)
                acc_list.append(acc)
                print(
                    'The accuracies for the predictions with centroid 1 representing the normal and anomalous distributions respectively are ',
                    acc, 100 - acc)
            else:
                latent_vec = frozen_encoder(inp)
                mini_kmeans.partial_fit(latent_vec)

            if cnt % 8 == 0:
                cur_state = {'acc_list': acc_list, 'epoch': epoch}
                torch.save(cur_state, training_state_loc)

if __name__ == '__main__':
    main()