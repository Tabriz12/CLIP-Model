import torch
from torch.utils.data import DataLoader
from dataset import train_transform, test_transform
from dataset import Flickr8K_dataset
from config import CFG
from models.clip import CLIP
from torch.optim import AdamW, SGD
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os

writer = SummaryWriter("./experiments/")



def dynamic_padding(input_tuple, max_len):
    truncated_sample = []
    for tensor_dict in input_tuple:
        truncated_dict = {}
        for key, tensor in tensor_dict.items():
            truncated_tensor = tensor[:max_len]
            truncated_dict[key] = truncated_tensor
        truncated_sample.append(truncated_dict)
    return tuple(truncated_sample)


def collate_fn(batch):
    
    ims, captions  = zip(*batch)

    max_len = 0

    for tensor in captions:
        max_len = max(max_len, tensor['attention_mask'].sum().item())
    
    captions = dynamic_padding(captions, max_len)

    return ims, captions
    
    

def validate_one_epoch(data_loader, model):

    running_loss = 0.

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):

        ims = data[0]
        captions= data[1]
        

        loss = model(ims, captions)

        running_loss += loss.item()
            
    

    return running_loss/len(data_loader)




def train_one_epoch(data_loader, model, opt, epoch):
    
    running_loss = 0.
    last_loss = 0.
    for i, data in tqdm(enumerate(data_loader), total = len(data_loader)):

        ims = data[0]
        captions = data[1]
        
        opt.zero_grad()

        loss = model(ims, captions)

        loss.backward()


        # Adjust learning weights
        opt.step()

        running_loss += loss.item()
        print( loss.item())

        if i % 500 == 499:
            last_loss = running_loss / 499
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, global_step=epoch)
            
    writer.add_scalar('Loss', last_loss, global_step=epoch)

    return last_loss


def main():

    training_data = Flickr8K_dataset(split='train', transform=train_transform)
    testing_data = Flickr8K_dataset(split='test', transform=test_transform)

    train_loader = DataLoader(training_data, batch_size=CFG['batch_size'], shuffle=True, drop_last=True)#, collate_fn=collate_fn)
    test_loader = DataLoader(testing_data, batch_size=CFG['batch_size']) #
    
    model = CLIP().to(CFG['device'])

    #for name, param in model.named_parameters():
     #   if not param.requires_grad:
      #      print (name, param.data)

    parameters =  [
        {"params": model.vision_model.parameters(), "lr": CFG['loss']['vision']},
        {"params": model.text_model.parameters(), "lr": CFG['loss']['text']},
        {"params": model.img_embbedder.parameters(), "lr": CFG['loss']['mapper']},
        {"params": model.txt_embedder.parameters(), "lr": CFG['loss']['mapper']},       
    ]

    print(parameters)
    

    

    optimizer = SGD(parameters)

    last_loss = 10000

    for e in range(CFG['epoch']):

        model.train()

        train_one_epoch(train_loader, model, optimizer, e)

        model.eval()

        avg_loss = validate_one_epoch(test_loader, model)

        print("Mean loss in validation data: {}".format(avg_loss))

        if avg_loss < last_loss:
            print("Saving the model")
            os.makedirs("./checkpoints/", exist_ok=True)
            torch.save(model.state_dict(), "best.pt")




if __name__ == "__main__":
    main()
