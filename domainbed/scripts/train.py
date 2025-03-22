# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

import argparse
import collections
import json
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import copy
import torch.nn as nn

#from torch.utils.tensorboard import SummaryWriter


from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import model_selection
from domainbed.lib.query import Q
from torchvision import transforms

import os

def load_algorithm(file_path,hparams):

    print("Filepath #####",file_path)
    d = torch.load(file_path)
    args = d["args"]
    print("Saved Model ++++++++++++++++++++++")
    # print('Saved Args:')
    # for k in args:
    #     print('\t{}: {}'.format(k, args[k]))

    #hparams=d["model_hparams"]
    hparams["data_augmentation"] = False
    hparams["eval"]=True
    #hparams['empty_head']=False
    
    # print('Saved HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))
 
    algo = algorithms.get_algorithm_class(d["args"]["algorithm"])
    model = algo(
        input_shape=d["model_input_shape"], 
        num_classes=d["model_num_classes"], 
        num_domains=d["model_num_domains"], 
        hparams=hparams
    )
    for i in d["model_dict"]:
        if "rand_conv" in i:
            del d["model_dict"][i]
        # if "classifier" in i:
        #     print("Classifier in saved model",i)
    for j in model.state_dict():
        if "rand_conv" in j:
            del model.state_dict()[j]
        # if "classifier" in j:
        #     print("Classifier in loaded template model",j)
        
    
    model.load_state_dict(d["model_dict"])
    hparams["eval"]=False

    
    
    model.train()

    return model, args, hparams

def ME_ADA_AUGMENT(in_splits, algorithm, device, N_WORKERS, hparams, args):

    if len(in_splits) != 1:
        raise ValueError("The list must contain exactly one element.")
    
    # if not hparams["custom_train_val"]:
    #     train_data = in_splits[0][0].underlying_dataset
    # else:
    train_data = in_splits[0][0]

    default_transform = train_data.transform

    test_transform = transforms.Compose(
                [transforms.Resize(224,antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(hparams['mean_std'][0],hparams['mean_std'][1])])

    train_data.transform = test_transform
    
    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=hparams['batch_size'],
            shuffle=False,
            num_workers=N_WORKERS)
    images, labels = [], []
    recover_transform = transforms.Compose([
                        misc.Denormalise(hparams['mean_std'][0],hparams['mean_std'][1]),
                        misc.Clamp(min_val=0, max_val=1)])
    
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(227,antialias=True)
    ])
    
    algorithm.eval()
    print("Creating Augmented Images >>>>>>>")
    for images_train, labels_train in train_loader:
        
        # wrap the inputs and labels in Variable
        images_train, labels_train = images_train.to(device), labels_train.to(device)
        inputs1, targets1 = algorithm.maximize(images_train, labels_train)
        images += [image_transform(x) for x in torch.unbind(recover_transform(inputs1), dim=0)]
        labels += [x.item() for x in torch.unbind(targets1,dim=0)]
    
    images = np.stack(images)

   
    in_splits[0][0].data = np.concatenate([in_splits[0][0].data,images])
    in_splits[0][0].targets.extend(labels)
    print("New dataset size ",len(in_splits[0][0].data))
    in_splits[0][0].transform = default_transform

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    
    train_minibatches_iterator = zip(*train_loaders)
    algorithm.train() 

    return in_splits, train_minibatches_iterator

def ME_ADA_STEP(in_splits, epoch, final_epoch, batch_size, step):

    unfinished_epochs = final_epoch-(epoch+1)
    if unfinished_epochs != 0:
        steps_per_epoch = round(min([len(env) / batch_size for env, _ in in_splits]))
        n_steps = unfinished_epochs*steps_per_epoch+step
    return n_steps, steps_per_epoch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="ViT_RB_small")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="test_ViT_RB")
    parser.add_argument('--continue_checkpoint', type=str, default=" ")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    


    
    args = parser.parse_args()
    args.save_best_model = True
    #writer = SummaryWriter(comment=args.output_dir.split("/")[-2])
    
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        js = json.loads(args.hparams)
        js["test_env"] = args.test_envs
        # print(args.hparams)
        hparams.update(js)
    
    if args.continue_checkpoint != " ":
        hparams["continue_checkpoint"] = args.continue_checkpoint
    
    if hparams["continue_checkpoint"] != " ":
        print("Filepath",hparams["continue_checkpoint"])
        print("Loading model details")
        model, _, _ = load_algorithm(hparams["continue_checkpoint"],hparams)
        
    

    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
        #torch.cuda.set_device("cuda:2")
        #torch.cuda.set_device(2)
    else:
        device = "cpu"

    print('device:', device)
    print ('Current cuda device ', torch.cuda.current_device())

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError
    
    hparams["total_steps"] = dataset.N_STEPS

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    if hparams["continue_checkpoint"] == " ":
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                    len(dataset) - len(args.test_envs), hparams)

        if algorithm_dict is not None:
            algorithm.load_state_dict(algorithm_dict)
    else:
        print("Saved model loaded")
        algorithm = model

    
    algorithm.to(device)

    in_splits = []
    out_splits = []
    uda_splits = []
    assigned = False
    for env_i, env in enumerate(dataset):  # env is a domain
        uda = []
        in_ = []
        out = []
    
        if hparams['custom_train_val'] and not (hparams['custom_val'] == env_i or hparams['custom_train'] == env_i):
            # out, in_ = misc.split_dataset(env,
            #                           int(len(env) * args.holdout_fraction),
            #                           misc.seed_hash(args.trial_seed, env_i))
            out = env
            
        elif  hparams['custom_train_val'] and (hparams['custom_val'] == env_i or hparams['custom_train'] == env_i):
            if not assigned:
                for env_j, env_ in enumerate(dataset):
                
                    if  hparams['custom_val'] == env_j:
                        out = env_
                    elif  hparams['custom_train'] == env_j:
                        in_= env_
                assigned = True
                print(len(out)," length of val ",len(in_)," length of train") 
            else:
                continue  
        else:
            if env_i in args.test_envs:
                out = env
            else:
                if args.dataset == "PACS_Custom":
                    out, in_ = misc.split_dataset_PACS_Custom(env,
                                        int(len(env) * args.holdout_fraction),
                                        misc.seed_hash(args.trial_seed, env_i))
                # elif args.dataset == "DIGITS":
                #     in_ = env
                else:
                    out, in_ = misc.split_dataset(env,
                                        int(len(env) * args.holdout_fraction),
                                        misc.seed_hash(args.trial_seed, env_i))
                    


        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        if len(in_)!=0:
            in_splits.append((in_, in_weights))
        if len(out)!=0:
            out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")
    envs_d = datasets.get_dataset_class(args.dataset).ENVIRONMENTS
    # for i in range(len(in_splits)):
    #     print("env ",envs_d[i]," in ",len(in_splits[i][0])," out ",len(out_splits[i][0]))
    for i in range(len(envs_d)):
        if hparams['custom_train_val']:
            if i ==hparams['custom_train']:
                print("env ",i," : ",envs_d[i]," in ",len(in_splits[i][0]))
            else:
                print("env ",i," : ",envs_d[i]," out ",len(out_splits[i-1][0]))
        else:
            
            if i ==hparams['custom_train']:
                print("env ",i," : ",envs_d[i]," in ",len(in_splits[i][0])," out ",len(out_splits[i][0]))
            else:
                print("env ",i," : ",envs_d[i]," out ",len(out_splits[i][0]))
    

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    # eval_loaders = [FastDataLoader(
    #     dataset=env,
    #     batch_size=64,
    #     num_workers=dataset.N_WORKERS)
    #     for env, _ in (in_splits + out_splits + uda_splits)]
    if hparams['custom_train_val']:
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=128,
            num_workers=dataset.N_WORKERS)
            for env, _ in (out_splits + uda_splits)]
    else:
        eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=128,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    
    # #eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    if hparams['custom_train_val']:
        eval_weights = [None for _, weights in (out_splits + uda_splits)]
    else:
        eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    # eval_loader_names = ['env{}_in'.format(i)
    #                      for i in range(len(in_splits))]
    if hparams['custom_train_val']:
        eval_loader_names = ['env{}_out'.format(i+1)
                            for i in range(len(out_splits))]
    else:
        eval_loader_names = ['env{}_in'.format(i)
                          for i in range(len(in_splits))]
        eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    
    print(len(train_loaders)," length of train loader ++++++++++++++++")
    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    if args.dataset == "DIGITS" or args.dataset == "PACS":
        print("steps_per_epoch ",dataset.STEPS_PER_EPOCH)
        steps_per_epoch = dataset.STEPS_PER_EPOCH
    else:   
        steps_per_epoch = round(min([len(env) / hparams['batch_size'] for env, _ in in_splits]))

    if hparams["continue_checkpoint"] == " ":
        start_step = 0
    else:
        start_step=hparams["checkpoint_step_start"]

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    print(f"+ checkpoint_freq: {checkpoint_freq}")


    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict(),
            
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    def save_checkpoint_best(filename, algo):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algo.state_dict(),
            
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))



    last_results_keys = None
    best_val_acc = 0
    step = start_step
    epoch = 0
    final_epoch = round(n_steps/steps_per_epoch)
    ME_ADA_k = 0

    adverserial_algorithms = ["ME_ADA_ViT","ME_ADA_CNN","ADA_CNN","ADA_ViT"]
    
    while(step!=n_steps):

        #TODO - add a parameter to control this 
        # step wise increase the inconsistancy loss
        # if step % 40000 ==0:  
        #     print("Increase consistency loss **********")         
        #     hparams['invariant_loss']=True
        #     hparams['consistency_loss_w']+=2
        
        if (step!=0) and (step%steps_per_epoch==0):
            epoch+=1
            hparams["epoch"] = epoch

            # if args.algorithm == "ALT_CNN":
            #     algorithm.network.train()
            #     algorithm.trans_module.eval()


            if (args.algorithm in adverserial_algorithms) and ((epoch+1)%hparams["epochs_min"]==0) and (ME_ADA_k<hparams["k"]):
                print("Augmenting the Dataset >>>>>>>>>>")
                in_splits, train_minibatches_iterator = ME_ADA_AUGMENT(in_splits, algorithm,device, dataset.N_WORKERS, hparams, args)
                n_steps, steps_per_epoch = ME_ADA_STEP(in_splits, epoch, final_epoch, hparams['batch_size'], step)
                print("Total steps ",n_steps," Steps per Epoch ",steps_per_epoch)
                ME_ADA_k+=1
                checkpoint_freq = checkpoint_freq*2
                hparams["total_steps"] = n_steps
                if hparams['scheduler']:
                    algorithm.scheduler.T_max = hparams["total_steps"]

        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                          for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = device
        
        step_vals = algorithm.update(minibatches_device, uda_device)
        
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        # print("Training done")
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # Training = []
            # for m in algorithm.network.modules():
            #     if isinstance(m, torch.nn.BatchNorm2d):
            #         Training.append(m.training)

            # check_True=all(element == True for element in Training)

            # if check_True:
            #     print ("Batch Norm layers are training")
            # else:
            #     print ("Batch Norm layers are NOT training")


            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            temp_acc = 0
            temp_count = 0
            
            for name, loader, weights in evals:
                acc,loss = misc.accuracy(algorithm, loader, weights, device,val_id=hparams['custom_val'],current_id=int(name[3]),randconv=hparams['val_augmentation'])
                if args.save_best_model:
                    #if int(name[3]) not in args.test_envs and "out" in name:
                    if hparams['custom_val'] == int(name[3]) and "out" in name:
                        temp_acc += acc
                        temp_count += 1
                results[name + '_acc'] = acc
                results[name + '_loss'] = loss
            
            # print("Validation done")    
            if args.save_best_model:
                val_acc = temp_acc / (temp_count * 1.0)
                if val_acc >= best_val_acc:
                    # model_save = algorithm.detach().clone()  # clone
                    # model_save = copy.deepcopy(algorithm)  # clone
                    if (args.save_best_model):
                        save_checkpoint('IID_best.pkl')
                        algorithm.to(device)
                    best_val_acc = val_acc
                    print("Best model upto now")
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            # writer.add_scalar("val_acc",results["env1_out_acc"],results["step"])
            # writer.add_scalar("val_loss",results["env1_out_loss"],results["step"])
            
            # if "task_loss" in list(results.keys()):
            #     writer.add_scalar("train_loss",results["task_loss"],results["step"])
            #     writer.add_scalar("inv_loss",results["inv_loss"],results["step"])
            # else:
            #     writer.add_scalar("train_loss",results["loss"],results["step"])
            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])
            
            
            # records = []
            # with open(epochs_path, 'r') as f:
            #     for line in f:
            #         records.append(json.loads(line[:-1]))
            # records = Q(records)
            # scores = records.map(model_selection.IIDAccuracySelectionMethod._step_acc)
            # if scores[-1] == scores.argmax('val_acc'):
            #     save_checkpoint('IID_best.pkl')
            #     algorithm.to(device)
            save_checkpoint(f'model_step_last.pkl')
            algorithm.to(device)
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

        step+=1
        # print("One iteration done")

    save_checkpoint('model.pkl')
    #writer.close()
    # if (args.save_best_model):
    #     save_checkpoint_best('IID_best.pkl', model_save)
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
