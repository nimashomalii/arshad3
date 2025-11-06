from dataset.main import data , data_for_subject_dependet
import torch 
import os # os را برای چک کردن cuda اضافه کنید
from models_structures.hippoLegS1 import model
from train import Trainer
import random
from functions import k_fold_data_segmentation
from  torch.utils.data import DataLoader , TensorDataset
import numpy as np 
import torch.nn as nn
#____Model______#
def create_model(test_person , emotion,category , fold_idx ) : 
    overlap = 0
    time_len = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    batch_size = 64
    data_type = torch.float32
    my_dataset = data(test_person, overlap, time_len, device, emotion, category, batch_size, data_type)
    train_loader = my_dataset.train_data()
    test_loader = my_dataset.test_data()

    x_dim , h_dim , seq_len ,c_dim = 14 , 32, 128*time_len, 40
    dim2 , dim3  = 64 , 16 
    Model = model( x_dim, h_dim, c_dim   ,seq_len,dim2 , dim3 , output_dim)# معماری دلخواه
    # class weights for imbalance
    y_train = my_dataset.y_train
    class_count = torch.bincount(y_train.long())
    class_count = class_count + (class_count == 0).long()
    weights = (1.0 / class_count.float())
    weights = weights / weights.sum() * len(class_count)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    #____trainer_______#
    trainer = Trainer(
        model=Model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        label_method=category,
        optimizer_cls=torch.optim.Adam,
        lr=5e-5,
        epochs=25,
        loss_fn = criterion ,
        checkpoint_path=f"eeg_checkpoint{fold_idx }.pth",
        log_path=f"eeg_log{fold_idx }.json", 
    )
    #____fit_model_____#
    return  trainer.fit()

def subject_dependent_validation (emotion ,category, fold_idx , k=5) : 
    overlap = 0
    time_len = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if category == 'binary'  :
        output_dim = 2 
    elif category == '5category' :
        output_dim = 5
    batch_size = 32
    data_type = torch.float32
    accuracies_on_subjects  = {
        'train' : [] , 
        'test' : []
    } 
    person_num =0
    data = data_for_subject_dependet(overlap , time_len ,emotion ,category ,data_type , device , k  )
    for (x , y) in data.data : 
        fold_idx = 0
        len_data = x.shape[0]
        fold_number = len_data//k 
        all_x = [x[fold_number*i : min(fold_number*(i+1) , len_data) , : , : ] for i in range(k)]
        all_y = [y[fold_number*i : min(fold_number*(i+1) , len_data)] for i in range(k)]
        print("\n" + "="*60)
        print(f"Subject {person_num}: Training {k}-fold cross-validation")
        print("="*60)
        for i in range(k): 
            print(f"\n-- Fold {i+1}/{k} --")
            x_test = all_x[i]
            y_test = all_y[i]
            x_train = all_x[:i] + all_x[i+1:]
            y_train = all_y[:i] + all_y[i+1:]
            x_train = torch.concat(x_train , dim=0)
            y_train = torch.concat(y_train , dim=0)

            test_dataset = TensorDataset(x_test , y_test)
            test_loader = DataLoader(test_dataset ,batch_size , shuffle=False)
            train_dataset = TensorDataset(x_train , y_train )
            train_loader = DataLoader(train_dataset , batch_size,shuffle=True )
            x_dim , h_dim , seq_len ,c_dim = 14 , 32 , 128*time_len, 32
            dim2 , dim3  = 64 , 16 
            Model = model( x_dim, h_dim, c_dim   ,seq_len,dim2 , dim3 , output_dim)# معماری دلخواه
            criterion = nn.CrossEntropyLoss()

            #____trainer_______#
            trainer = Trainer(
                model=Model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                label_method=category,
                optimizer_cls=torch.optim.Adam,
                lr=4e-5,
                epochs=30,
                loss_fn = criterion ,
                verbose=True,
                save_each_epoch=False,
                checkpoint_path=f"eeg_checkpoint{fold_idx + person_num*5}.pth",
                log_path=f"eeg_log{fold_idx + person_num*5}.json", 
            )
            #____fit_model_____#
            history =  trainer.fit()
            fold_train_acc = np.mean(np.array(history['train_acc'][-5:]))
            fold_val_acc = np.mean(np.array(history['val_acc'][-5:]))
            print(f"Fold {i+1} result -> Train Acc (last5 avg): {fold_train_acc:.2f}% | Test Acc (last5 avg): {fold_val_acc:.2f}%")
            if fold_idx ==0 : 
                train_loss = np.array(history['train_loss'])
                val_loss = np.array(history['val_loss'])
                train_acc = np.array(history['train_acc'])
                val_acc = np.array(history['val_acc'])
            else : 
                train_loss += np.array(history['train_loss'])
                val_loss += np.array(history['val_loss'])
                train_acc += np.array(history['train_acc'])
                val_acc += np.array(history['val_acc'])
            fold_idx +=1
        person_num +=1
        train_acc  /=k
        train_loss /=k
        val_loss   /=k
        val_acc    /=k

        accuracies_on_subjects['train'].append(np.mean(np.array(train_acc[-5:])))
        accuracies_on_subjects['test'].append(np.mean(np.array(val_acc[-5:])))
    return accuracies_on_subjects



