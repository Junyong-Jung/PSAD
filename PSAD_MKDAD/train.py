from test import *
from utils.utils import *
from dataloader import *
from pathlib import Path
from torch.autograd import Variable
import pickle
from test_functions import detection_test
from loss_functions import *
import random
parser = ArgumentParser()
# parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--category',type=str, required=True)

class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score,
                                                                               self.max_epoch))
        
def train(config):
    score_obs = Score_Observer('AUROC')    
    direction_loss_only = config["direction_loss_only"]
    normal_class = config["normal_class"]
    learning_rate = float(config['learning_rate'])
    num_epochs = config["num_epochs"]
    lamda = config['lamda']
    continue_train = config['continue_train']
    last_checkpoint = config['last_checkpoint']

    checkpoint_path = "./outputs/{}/{}/checkpoints/".format(config['experiment_name'], config['dataset_name'])

    # create directory
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader = load_data(config)
    if continue_train:
        vgg, model = get_networks(config, load_checkpoint=True)
    else:
        vgg, model = get_networks(config)

    # Criteria And Optimizers
    if direction_loss_only:
        criterion = DirectionOnlyLoss()
    else:
        criterion = MseDirectionLoss(lamda)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if continue_train:
        optimizer.load_state_dict(
            torch.load('{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, last_checkpoint)))

    losses = []
    roc_aucs = []
    if continue_train:
        with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, last_checkpoint), 'rb') as f:
            roc_aucs = pickle.load(f)

    for epoch in range(num_epochs + 1):
        model.train()
        epoch_loss = 0
        for data in train_dataloader:
            X = data[0]
            if X.shape[1] == 1:
                X = X.repeat(1, 3, 1, 1)
            X = Variable(X).cuda()

            output_pred = model.forward(X)
            output_real = vgg(X)

            total_loss = criterion(output_pred, output_real)

            # Add loss to the list
            epoch_loss += total_loss.item()
            losses.append(total_loss.item())

            # Clear the previous gradients
            optimizer.zero_grad()
            # Compute gradients
            total_loss.backward()
            # Adjust weights
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        if epoch % 10 == 0:
            roc_auc = detection_test(model, vgg, test_dataloader, config)
            roc_aucs.append(roc_auc)
            score_obs.update(roc_auc,epoch,print_score=True)

        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       '{}Cloner_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            torch.save(optimizer.state_dict(),
                       '{}Opt_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, epoch))
            with open('{}Auc_{}_epoch_{}.pickle'.format(checkpoint_path, normal_class, epoch),
                      'wb') as f:
                pickle.dump(roc_aucs, f)


def main():
    args = parser.parse_args()
    config = get_config('configs/config.yaml')
    config['normal_class'] = args.category
    train(config)


if __name__ == '__main__':
    random.seed(100)
    main()
