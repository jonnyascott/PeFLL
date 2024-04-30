import argparse
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
from tqdm import trange

from models import CNNHyper, CNNTarget, MLPEmbed, EmbedHyper, MLP, CNNEmbed
from node import BaseNodes
from utils import get_device, set_logger, set_seed, str2bool


def eval_model(nodes, train_idxs, test_idxs, joint, net, criteria, device, m, s, split, embed_split,
               mask_absent_classes=False):
    num_nodes = len(train_idxs) + len(test_idxs)
    curr_results, embeddings = evaluate(nodes, num_nodes, joint, net, criteria, device, m, s, split=split, embed_split=embed_split,
                            mask_absent_classes=mask_absent_classes)

    results = dict()
    l1, l2 = [train_idxs], ['train_nodes']
    if test_idxs:
        l1.append(test_idxs)
        l2.append('test_nodes')

    for idxs, key in zip(l1, l2):
        total_correct = sum([curr_results[i]['correct'] for i in idxs])
        total_samples = sum([curr_results[i]['total'] for i in idxs])
        avg_loss = np.mean([curr_results[i]['loss'] for i in idxs])
        avg_acc = total_correct / total_samples
        all_acc = [curr_results[i]['correct'] / curr_results[i]['total'] for i in idxs]

        # embeddings = [curr_results[i]['embedding'] for i in idxs]

        results[key] = dict(zip(["avg_loss", "avg_acc", "all_acc"], [avg_loss, avg_acc, all_acc]))

    return results, embeddings


@torch.no_grad()
def evaluate(nodes: BaseNodes, num_nodes, joint, net, criteria, device, m, s, split='test', embed_split='train',
             mask_absent_classes=False):
    joint.eval()
    results = defaultdict(lambda: defaultdict(list))
    embeddings = []
    for node_id in range(num_nodes):  # iterating over nodes
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        if embed_split == 'train':
            dl = nodes.train_loaders[node_id]
        else:
            dl = curr_data

        num_batches = len(dl)
        embedding = 0.
        l = 0
        classes_present = 0.
        for i, B in enumerate(dl):
            l += len(B)
            B = tuple(t.to(device) for t in B)
            _, y = B
            classes_present += y.sum(0)
            embedding += joint.embednet(B).sum(0)
            if i + 1 == num_batches:
                break

        classes_present = classes_present >= 1

        embedding = embedding / l
        embedding = (embedding - m) / s

        embeddings.append(embedding.cpu().detach().numpy())

        weights = joint.hypernet(embedding)
        net.load_state_dict(weights)

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)
            pred = net(img)
            if mask_absent_classes:
                pred = pred * classes_present
            running_loss += criteria(pred, label).item()
            running_correct += pred.argmax(1).eq(label.argmax(1)).sum().item()
            running_samples += len(label)

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

    embeddings = np.array(embeddings)

    return results, embeddings


def finetune(net, optim, dls, epochs, criteria, device):
    train_dl, val_dl, test_dl = dls
    val_losses, val_accuracies = [], []
    test_losses, test_accuracies = [], []
    for epoch in range(epochs + 1):
        if epoch > 0:
            net.train()
            for batch in train_dl:
                img, label = tuple(t.to(device) for t in batch)
                pred = net(img)
                loss = criteria(pred, label)

                optim.zero_grad()
                loss.backward()
                optim.step()

        running_loss, running_correct, running_samples = 0., 0., 0.
        with torch.no_grad():
            net.eval()
            for batch_count, batch in enumerate(val_dl):
                img, label = tuple(t.to(device) for t in batch)
                pred = net(img)
                running_loss += criteria(pred, label).item()
                running_correct += pred.argmax(1).eq(label.argmax(1)).sum().item()
                running_samples += len(label)

            val_losses.append(running_loss / (batch_count + 1))
            val_accuracies.append(running_correct / running_samples)

        running_loss, running_correct, running_samples = 0., 0., 0.
        with torch.no_grad():
            net.eval()
            for batch_count, batch in enumerate(test_dl):
                img, label = tuple(t.to(device) for t in batch)
                pred = net(img)
                running_loss += criteria(pred, label).item()
                running_correct += pred.argmax(1).eq(label.argmax(1)).sum().item()
                running_samples += len(label)

            test_losses.append(running_loss / (batch_count + 1))
            test_accuracies.append(running_correct / running_samples)

    return test_losses, test_accuracies, val_losses, val_accuracies


def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int, num_train_nodes: int,
          clients_per_step: int, partition_type: str, alpha_train: float, alpha_test: float,
          steps: int, inner_steps: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, embed_hid: int, embed_nlayers: int,
          embed_batches: int, embed_split: str, embed_y: bool, embed_model: str, hyper_hid: int, hyper_nhid: int,
          n_kernels: int, bs: int, device, eval_every: int, save_path: Path, mask_absent: bool, seed: int) -> None:

    ###############################
    # init nodes, hnet, local net #
    ###############################
    alpha_test_range = None

    all_embeddings = []
    embedding_dir_path = None
    # Infer on range of OOD test clients
    if alpha_test == -1:
        assert partition_type == 'dirichlet'
        alpha_test_range = np.arange(1, 11) * 0.1
        alpha_test = alpha_train

    if data_name == 'femnist':
        num_nodes = 3597
        num_train_nodes = int(0.9 * num_nodes)

    nodes = BaseNodes(data_name, data_path, num_nodes, num_train_nodes, partition_type=partition_type,
                      classes_per_node=classes_per_node, batch_size=bs, alpha_train=alpha_train, alpha_test=alpha_test,
                      embedding_dir_path=embedding_dir_path)

    train_idxs = list(range(num_train_nodes))
    test_idxs = list(range(num_train_nodes, num_nodes))

    embed_dim = embed_dim
    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + num_nodes / 4)

    if data_name == "cifar10":
        embed_x = True
        dim_x = 32 * 32 * 3
        dim_y = 10
        if embed_model == 'mlp':
            enet = MLPEmbed(10, embed_dim)
        elif embed_model == 'cnn':
            enet = CNNEmbed(embed_y, 10, embed_dim, device)
        else:
            raise ValueError('Choose model from mlp or cnn.')
        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid, n_hidden=hyper_nhid, n_kernels=n_kernels)
        joint = EmbedHyper(enet, hnet)
        net = CNNTarget(n_kernels=n_kernels)
    elif data_name == "cifar100":
        if embed_model == 'mlp':
            enet = MLPEmbed(100, embed_dim)
        elif embed_model == 'cnn':
            enet = CNNEmbed(embed_y, 100, embed_dim, device)
        else:
            raise ValueError('Choose model from mlp or cnn.')

        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid,
                        n_hidden=hyper_nhid, n_kernels=n_kernels, out_dim=100)
        joint = EmbedHyper(enet, hnet)
        net = CNNTarget(n_kernels=n_kernels, out_dim=100)
    elif data_name == 'femnist':
        if embed_model == 'mlp':
            enet = MLPEmbed(62, embed_dim)
        elif embed_model == 'cnn':
            enet = CNNEmbed(embed_y, 62, embed_dim, device, in_channels=1)
        else:
            raise ValueError('Choose model from mlp or cnn.')

        hnet = CNNHyper(num_nodes, embed_dim, in_channels=1, hidden_dim=hyper_hid,
                        n_hidden=hyper_nhid, n_kernels=n_kernels, out_dim=62)
        joint = EmbedHyper(enet, hnet)
        net = CNNTarget(in_channels=1, n_kernels=n_kernels, out_dim=62)

    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

    joint = joint.to(device)
    net = net.to(device)

    filename = f"{data_name}_{num_nodes}_nodes_{num_train_nodes}_trainnodes_" \
               f"_partition_{partition_type}_alphatrain_{alpha_train}_alphatest_{alpha_test}" \
               f"_seed_{seed}"

    checkpoint_dir = Path(f'saved_models/{filename}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ##################
    # init optimizer #
    ##################
    embed_lr = embed_lr if embed_lr is not None else lr

    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in joint.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in joint.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr, momentum=0.9, weight_decay=wd
        ),
        'adam': torch.optim.Adam(params=joint.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)

    results = {'train_nodes': defaultdict(list), 'test_nodes': defaultdict(list)}
    m, s = None, None
    already_embedded = False
    for step in step_iter:
        if (step == 1) and (embed_batches == -1):
            print('Using full client dl to generate embedding')
        joint.train()

        # select client at random
        node_ids = random.sample(train_idxs, clients_per_step)

        all_grads = []
        for node_id in node_ids:
            # produce & load local network weights
            dl = nodes.train_loaders[node_id]
            num_batches = embed_batches if (embed_batches != -1) else len(dl)
            embedding = torch.zeros(embed_dim).to(device)
            l = 0
            for i, B in enumerate(dl):
                l += len(B)
                B = tuple(t.to(device) for t in B)
                embedding += enet(B).sum(0)
                if i + 1 == num_batches:
                    break

            embedding = embedding / l

            if m is None:
                with torch.no_grad():
                    m, s = torch.mean(embedding), torch.std(embedding)

            embedding = (embedding - m) / s

            if step == 0 and not already_embedded and embedding_dir_path is not None:
                _, embeddings = eval_model(nodes, train_idxs, test_idxs, joint, net, criteria,
                                                      device, m, s, split="test", embed_split=embed_split,
                                           mask_absent_classes=mask_absent)
                all_embeddings.append(embeddings)
                if embedding_dir_path is not None:
                    np.save(f'{embedding_dir_path}/user_embeddings.npy', np.array(all_embeddings))
                already_embedded = True

            weights = hnet(embedding)
            net.load_state_dict(weights)

            # init inner optimizer
            inner_optim = torch.optim.SGD(
                net.parameters(), lr=inner_lr, momentum=.9, weight_decay=inner_wd
            )

            # storing theta_i for later calculating delta theta
            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            # NOTE: evaluation on sent model
            with torch.no_grad():
                net.eval()
                batch = next(iter(nodes.test_loaders[node_id]))
                img, label = tuple(t.to(device) for t in batch)
                pred = net(img)
                prvs_loss = criteria(pred, label)
                prvs_acc = pred.argmax(1).eq(label.argmax(1)).sum().item() / len(label)
                net.train()

            # inner updates -> obtaining theta_tilda
            for i in range(inner_steps):
                net.train()
                inner_optim.zero_grad()
                optimizer.zero_grad()

                batch = next(iter(nodes.train_loaders[node_id]))
                img, label = tuple(t.to(device) for t in batch)
                pred = net(img)
                loss = criteria(pred, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

                inner_optim.step()

            optimizer.zero_grad()

            final_state = net.state_dict()

            # calculating delta theta
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            # calculating phi gradient
            joint_grads = torch.autograd.grad(
                list(weights.values()), joint.parameters(), grad_outputs=list(delta_theta.values()), allow_unused=True
            )

            all_grads.append(joint_grads)

        sum_grads = [0. for _ in range(len(all_grads[0]))]

        for g in all_grads:
            sum_grads = [s_i + g_i for (s_i, g_i) in zip(sum_grads, g)]

        avg_grads = [s_i / clients_per_step for s_i in sum_grads]
        # update hnet weights
        for p, g in zip(joint.parameters(), avg_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(joint.parameters(), 50)
        optimizer.step()

        step_iter.set_description(
            f"Step: {step+1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_acc:.4f}"
        )

        if step % eval_every == 0:
            last_eval = step
            step_results, embeddings = eval_model(nodes, train_idxs, test_idxs, joint, net, criteria,
                                      device, m, s, split="test", embed_split=embed_split, mask_absent_classes=mask_absent)

            all_embeddings.append(embeddings)
            if embedding_dir_path is not None:
                np.save(f'{embedding_dir_path}/user_embeddings.npy', np.array(all_embeddings))

            avg_acc = step_results['train_nodes']['avg_acc']
            avg_loss = step_results['train_nodes']['avg_loss']
            all_acc = step_results['train_nodes']['all_acc']

            logging.info(f"\nStep: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

            for key, dic in step_results.items():
                results[key]['test_avg_loss'].append(dic['avg_loss'])
                results[key]['test_avg_acc'].append(dic['avg_acc'])

            step_val_results, _ = eval_model(nodes, train_idxs, test_idxs, joint, net, criteria,
                                          device, m, s, split="val", embed_split=embed_split, mask_absent_classes=mask_absent)

            val_avg_loss = step_val_results['train_nodes']['avg_loss'],
            val_avg_acc = step_val_results['train_nodes']['avg_acc']

            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            results['train_nodes']['val_avg_loss'].append(val_avg_loss)
            results['train_nodes']['val_avg_acc'].append(val_avg_acc)
            results['train_nodes']['best_step'].append(best_step)
            results['train_nodes']['best_val_acc'].append(best_acc)
            results['train_nodes']['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
            results['train_nodes']['test_best_min_based_on_step'].append(test_best_min_based_on_step)
            results['train_nodes']['test_best_max_based_on_step'].append(test_best_max_based_on_step)
            results['train_nodes']['test_best_std_based_on_step'].append(test_best_std_based_on_step)

            torch.save({
                'step': step,
                'enet_state_dict': enet.state_dict(),
                'hnet_state_dict': hnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{checkpoint_dir}/step_{step}.ckpt')

    if step != last_eval:
        step_results, embeddings = eval_model(nodes, train_idxs, test_idxs, joint, net, criteria,
                                  device, m, s, split="test", embed_split=embed_split, mask_absent_classes=mask_absent)

        all_embeddings.append(embeddings)
        if embedding_dir_path is not None:
            np.save(f'{embedding_dir_path}/user_embeddings.npy', np.array(all_embeddings))

        avg_acc = step_results['train_nodes']['avg_acc']
        avg_loss = step_results['train_nodes']['avg_loss']
        all_acc = step_results['train_nodes']['all_acc']

        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

        for key, dic in step_results.items():
            results[key]['test_avg_loss'].append(dic['avg_loss'])
            results[key]['test_avg_acc'].append(dic['avg_acc'])

        step_val_results, _ = eval_model(nodes, train_idxs, test_idxs, joint, net, criteria,
                                      device, m, s, split="val", embed_split=embed_split,
                                         mask_absent_classes=mask_absent)

        val_avg_loss = step_val_results['train_nodes']['avg_loss'],
        val_avg_acc = step_val_results['train_nodes']['avg_acc']

        if best_acc < val_avg_acc:
            best_acc = val_avg_acc
            best_step = step
            test_best_based_on_step = avg_acc
            test_best_min_based_on_step = np.min(all_acc)
            test_best_max_based_on_step = np.max(all_acc)
            test_best_std_based_on_step = np.std(all_acc)

        results['train_nodes']['val_avg_loss'].append(val_avg_loss)
        results['train_nodes']['val_avg_acc'].append(val_avg_acc)
        results['train_nodes']['best_step'].append(best_step)
        results['train_nodes']['best_val_acc'].append(best_acc)
        results['train_nodes']['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
        results['train_nodes']['test_best_min_based_on_step'].append(test_best_min_based_on_step)
        results['train_nodes']['test_best_max_based_on_step'].append(test_best_max_based_on_step)
        results['train_nodes']['test_best_std_based_on_step'].append(test_best_std_based_on_step)

        torch.save({
            'step': step,
            'enet_state_dict': enet.state_dict(),
            'hnet_state_dict': hnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{checkpoint_dir}/step_{step}.ckpt')

    best_checkpoint = torch.load(f'{checkpoint_dir}/step_{best_step}.ckpt')

    enet.load_state_dict(best_checkpoint['enet_state_dict'])
    hnet.load_state_dict(best_checkpoint['hnet_state_dict'])

    # Infer on range of alpha clients
    if alpha_test_range is not None:
        results['ood_results'] = dict()
        for alpha_test_new in alpha_test_range:
            results['ood_results'][f'{alpha_test_new:.2f}'] = []
            print(f'Testing clients for alpha={alpha_test_new}')
            nodes = BaseNodes(data_name, data_path, num_nodes, num_train_nodes, partition_type=partition_type,
                              classes_per_node=classes_per_node, batch_size=bs, alpha_train=alpha_train,
                              alpha_test=alpha_test_new)

            ood_results, _ = eval_model(nodes, train_idxs, test_idxs, joint, net, criteria,
                                      device, m, s, split="test", embed_split=embed_split, mask_absent_classes=mask_absent)

            results['ood_results'][f'{alpha_test_new:.2f}'] = ood_results

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / filename) + '.json', "w") as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PeFLL Training Experiment Arguments"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100', 'femnist']
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for datasets")
    parser.add_argument("--num-nodes", type=int, default=100, help="number of simulated nodes")
    parser.add_argument("--num-train-nodes", type=int, default=90, help="number of nodes used in training")
    parser.add_argument("--clients-per-step", type=int, default=5, help="nodes to sample per round")
    parser.add_argument("--partition-type", type=str, default='by_class', help="[by_class, dirichlet]")
    parser.add_argument("--alpha-train", type=float, default=0.1, help="alpha for train clients")
    parser.add_argument("--alpha-test", type=float, default=0.1, help="alpha for test clients")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--optim", type=str, default='adam', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--hyper-nhid", type=int, default=3, help="num. hidden layers hypernetwork")
    parser.add_argument("--embed-nlayers", type=int, default=3, help="num. layers embedding network")
    parser.add_argument("--embed-batches", type=int, default=1, help="batches used to estimate rescaling")
    parser.add_argument("--embed-split", type=str, default='train', help="use train or test data to embed")
    parser.add_argument("--embed-y", type=str2bool, default=True, help="embed y as well as x")
    parser.add_argument("--embed-model", type=str, default='cnn', help="embed with cnn or mlp")
    parser.add_argument("--inner-lr", type=float, default=2e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--embed-hid", type=int, default=20, help="embednet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="results", help="dir path for output file")
    parser.add_argument("--mask-absent", type=str2bool, default=True, help="mask absent classes at eval")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    if args.gpu == -1:
        args.gpu = torch.cuda.current_device()

    device = get_device(gpus=args.gpu) if torch.cuda.is_available() else 'cpu'

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    else:
        args.classes_per_node = 10

    print('Running using seed:', args.seed)

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        num_train_nodes=args.num_train_nodes,
        clients_per_step=args.clients_per_step,
        partition_type=args.partition_type,
        alpha_train=args.alpha_train,
        alpha_test=args.alpha_test,
        steps=args.num_steps,
        inner_steps=args.inner_steps,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        embed_hid=args.embed_hid,
        embed_nlayers=args.embed_nlayers,
        embed_batches=args.embed_batches,
        embed_split=args.embed_split,
        embed_y=args.embed_y,
        embed_model=args.embed_model,
        hyper_hid=args.hyper_hid,
        hyper_nhid=args.hyper_nhid,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        mask_absent=args.mask_absent,
        seed=args.seed
    )
