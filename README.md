## Setups

- Python 3.6.4
- Pytorch 1.6.0

Make a folder to save models and download data (ISIC2019). Put them into the same directory as the `code` directory. 

```bash
mkdir models
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
```

Produce config files for a sub dataset with 5000 samples:

```bash
python split.py
# Please directly modificate the noise fraction in noisy_label.py
python noisy_label.py
```



## Baseline-clean

Setting:

- 5000 samples with clean labels
- ResNet 50
- Epoches: 200
- Batch Size: 32
- Learning Rate: 1e-3, decay every 50 epochs
- Augmentation: RandomRotation [-180, 180], RandomResize [0.3, 1], RandomHorizontalFlip 

command:
```bash
CUDA_VISIBEL_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 baseline_distributed.py --epochs=200 --fig-path=baseline > baseline.out &
```

Line 77 of baseline_distributed.py:
```bash
train = BasicDataset(dir_img, noise_fraction, mode='base')
```

## Baseline-0.2

Setting:

- 5000 samples
- Noise fraction 0.2: 4000 samples with clean labels and 1000 samples with noisy samples
- ResNet 50
- Epoches: 200
- Batch Size: 32
- Learning Rate: 1e-3, decay every 50 epochs
- Augmentation: RandomRotation [-180, 180], RandomResize [0.3, 1], RandomHorizontalFlip 

Command:
```bash
CUDA_VISIBEL_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 baseline_distributed.py --epochs=200 --noise-fraction=0.2 --fig-path=baseline_0.2 > baseline_0.2.out &
```

## Baseline-0.4

Setting:

- 5000 samples
- Noise fraction 0.4: 3000 samples with clean labels and 2000 samples with noisy samples
- ResNet 50
- Epoches: 200
- Batch Size: 32
- Learning Rate: 1e-3, decay every 50 epochs
- Augmentation: RandomRotation [-180, 180], RandomResize [0.3, 1], RandomHorizontalFlip 

Command:
```bash
CUDA_VISIBEL_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 baseline_distributed.py --epochs=200 --noise-fraction=0.4 --fig-path=baseline_0.4 > baseline_0.4.out &
```

## Learning to Reweight examples

Setting: 

- 5000 samples
- Noise fraction 0.4
- ResNet 50
- Epoches: 200
- Batch Size: 32
- Learning Rate: 1e-3, decay every 50 epochs
- Clean dataset: 5 or 125
- Augmentation: RandomRotation [-180, 180], RandomResize [0.3, 1], RandomHorizontalFlip 

Algorithms:

<center class="image">
<image src="pseudocode.PNG" width="375"/>
<image src="framework.png" width="375"/>
</center>


Implementation:
```bash
with higher.innerloop_ctx(net, opt) as (meta_net, meta_opt):
    # Line 4 in Algorithm1
    y_f_hat = meta_net(image)
    # Line 5 in Algorithm1
    cost = loss(y_f_hat, labels)
    eps = torch.zeros(cost.size()).cuda()
    eps = eps.requires_grad_()
    l_f_meta = torch.sum(cost * eps)
    # Line 6 and Line 7 in Algorithm1
    meta_opt.step(l_f_meta)
    # Line 8 in Algorithm1
    y_g_hat = meta_net(val_data)
    # Line 9 in Algorithm1
    l_g_meta = torch.mean(loss(y_g_hat, val_labels))
    # Line 10 in in Algorithm1
    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0].detach()
# Line 11 in Algorithm1
w_tilde = torch.clamp(-grad_eps, min=0)
norm_c = torch.sum(w_tilde) + 1e-10
if norm_c != 0:
    w = w_tilde / norm_c
else:
    w = w_tilde
# Line 12 in Algorithm1
y_f_hat = net(image)
cost = loss(y_f_hat, labels)
l_f = torch.sum(cost * w)
# Line 13 and Line 14 in Algorithm1
opt.zero_grad()
l_f.backward()
opt.step()
```

Modification: 

- Line 182
```bash
# w_tilde = torch.clamp(-grad_eps, min=0)
w_tilde = torch.sigmoid(-grad_eps)
```

- Line 210
```bash
nn.utils.clip_grad_norm_(net.parameters(), 0.25, norm_type=2)
```

Command:
```bash
CUDA_VISIBEL_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train_distributed.py --epochs=200 --noise-fraction=0.4 --fig-path=reweight_0.4 > reweight_0.4.out &
```

## Learning to produce mixup labels

Setting: 

- 5000 samples
- Noise fraction 0.4
- ResNet 50
- Epoches: 200
- Batch Size: 32
- Learning Rate: 1e-3, decay every 50 epochs
- Clean dataset: 5
- Augmentation: RandomRotation [-180, 180], RandomResize [0.3, 1], RandomHorizontalFlip 

We want to explore better strategies for label refurbishment in the meta-learning framework.

<center class = "label refurbishement">
<image src = "mixup.png" width = "400"/>
</center>

Implementation:

- Hard:
```bash
y_f_hat = net(image)
_, y_predicted = torch.max(y_f_hat, 1)
mixup_labels = beta * labels + (1-beta) * y_predicte
```


- Soft:
```bash
# Initialization before training process, Line 113
dict = {}
for i in range(len(data_loader)):
    _, labels, _, names = next(data)
    label = torch.zeros([batch_size, 9]).cuda(local_rank)
    for k in range(labels.shape[0]):
        label[k][int(labels[k])] = 1 
        dict[names[k]] = label[k]

...

# When training: Update mixup labels
y_f_hat = net(image)
prob = nn.functional.softmax(y_f_hat, dim=1).detach()
for k in range(labels.shape[0]):
    label[k] = dict[names[k]]
mixup_labels = beta * label + (1-beta) * prob
for k in range(labels.shape[0]):
    dict[names[k]] = mixup_labels[k]
```


We add another validation sub-network with the same structure as the one for learning to reweight examples for updating beta.


<center class = "label refurbishement">
<image src = "pseudo-label.png" width = "600"/>
</center>

Implementation:

```bash
with higher.innerloop_ctx(net, opt) as (meta_net, meta_opt):
    y_f_hat = meta_net(image)
    cost = loss(y_f_hat, labels)
    eps = torch.zeros(cost.size()).cuda(local_rank)
    eps = eps.requires_grad_()
    l_f_meta = torch.sum(cost * eps)
    meta_opt.step(l_f_meta)
    y_g_hat = meta_net(val_data)
    l_g_meta = torch.mean(loss(y_g_hat, val_labels))
    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, create_graph=True, retain_graph=True, allow_unused=True)[0].detach()

beta = torch.sigmoid(-grad_eps)
beta = beta.cuda(local_rank)
mixup_labels = HARD(beta) or SOFT(beta)
cost = loss(y_f_hat, mixup_labels.long())
w = torch.ones(cost.size()).cuda(local_rank)
l_f = torch.sum(cost * w)
opt.zero_grad()
l_f.backward()
nn.utils.clip_grad_norm_(net.parameters(), 0.25, norm_type=2)
opt.step()
```

Command:

- Hard:
```bash
CUDA_VISIBEL_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train_hard_distributed.py --epochs=200 --noise-fraction=0.4 --fig-path=mixup_hard_0.4 > mixup_hard_0.4.out &
```

- Soft
```bash
CUDA_VISIBEL_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train_soft_distributed.py --epochs=200 --noise-fraction=0.4 --fig-path=mixup_soft_0.4 > mixup_soft_0.4.out &
```



## Proposal: Learning to reweight examples & Learning to produce mixup labels


Setting: 

- 5000 samples
- Noise fraction 0.4
- ResNet 50
- Epoches: 200
- Batch Size: 32
- Learning Rate: 1e-3, decay every 50 epochs
- Clean dataset: 5
- Augmentation: RandomRotation [-180, 180], RandomResize [0.3, 1], RandomHorizontalFlip 

We have a sub-network for updating weight to reweight examples and add another sub-network to produce mixup labels. We update these two networks alternately.

<center class = "proposal">
<image src = "proposal.png" width = "600"/>
</center>


Upper Bound:

- Train a **Clean_Net** on 5000 clean data.
- Mark training samples with noisy labels as **1**, those with clean labels as **0**.
- Replace noisy labels with pseudo labels and remain clean labels.
- Optimize weights for each examples as in **Learning to reweight examples**. 
- Command:
```bash 
CUDA_VISIBEL_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 upperbound_distributed.py --epochs=200 --noise-fraction=0.4 --fig-path=upperbound_0.4 > upperbound_0.4.out &
```

- Implementation

```bash
y_clean = clean_net(image)
_, y_predicted_clean = torch.max(y_clean, 1)
# Soft: prob = nn.functional.softmax(y_clean, dim=1).detach()
for k in range(marks.shape[0]):
    # For examples with noisy labels
    if marks[k] == 1:
        # Hard: 
        mixup_labels[k] = y_predicted_clean[k]
        # Soft: mixup_labels[k] = prob[k]
    else:
    # For examples with clean labels
      mixup_labels[k] = labels[k]

cost = loss(y_f_hat, mixup_labels.long())
l_f = torch.sum(cost * w)
opt.zero_grad()
l_f.backward()
nn.utils.clip_grad_norm_(net.parameters(), 0.25, norm_type=2)
opt.step()
```


Implementation of Proposal:

```bash
if i % 2 != 0:
  # Learning to produce mixup labels
  with higher.innerloop_ctx(net, opt) as (meta_net, meta_opt):
      y_f_hat = meta_net(image)
      cost = loss(y_f_hat, labels)
      eps = torch.zeros(cost.size()).cuda(local_rank)
      eps = eps.requires_grad_()
      l_f_meta = torch.sum(cost * eps)
      meta_opt.step(l_f_meta)
      y_g_hat = meta_net(val_data)
      l_g_meta = torch.mean(loss(y_g_hat, val_labels))
      grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, create_graph=True, retain_graph=True, allow_unused=True)[0].detach()

  beta = torch.sigmoid(-grad_eps)
  beta = beta.cuda(local_rank)

if i % 2 == 0:
  # Learning to reweight examples
  with higher.innerloop_ctx(net, opt) as (meta_net, meta_opt):
      y_f_hat = meta_net(image)
      cost = loss(y_f_hat, labels)
      eps = torch.zeros(cost.size()).cuda()
      eps = eps.requires_grad_()
      l_f_meta = torch.sum(cost * eps)
      meta_opt.step(l_f_meta)
      y_g_hat = meta_net(val_data)
      l_g_meta = torch.mean(loss(y_g_hat, val_labels))
      grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0].detach()

  # w_tilde = torch.clamp(-grad_eps, min=0)
  w_tilde = torch.sigmoid(-grad_eps)
  norm_c = torch.sum(w_tilde)
  if norm_c != 0:
      w = w_tilde / norm_c
  else:
      w = w_tilde

y_f_hat = net(image)
mixup_labels = HARD(beta) or SOFT(beta)
cost = loss(y_f_hat, mixup_labels.long())
l_f = torch.sum(cost * w)
opt.zero_grad()
l_f.backward()
nn.utils.clip_grad_norm_(net.parameters(), 0.25, norm_type=2)
opt.step()
```
Command:
```bash
CUDA_VISIBEL_DEVICES=0,1,2,3 python -u -m torch.distributed.launch --nproc_per_node=4 train_mix_distributed.py --epochs=200 --noise-fraction=0.4 --fig-path=proposal_hard_0.4 > proposal_hard_0.4.out &
```


- Upper Bound:


## Results

**ISIC2019 ResNet**

| noise_fraction | 0.0 (clean) | 0.2         | 0.4         |
| :------------: | :---------: | :---------: | :---------: |
| Baseline       | 0.797       | 0.68        | 0.55        |
| Reweight - Clamp & Clean 5 | | 0.67        | 0.60        |
| Reweight - Sigmoid & Clean 5 | | 0.68      | 0.64        |
| Reweight - Sigmoid & Clean 125 | | 0.69    | 0.66        |
| Mixup Labels - Hard |        |             | 0.60        | 
| Mixup Labels - Soft |        |             | 0.56        |
| Proposal - Upper Bound |     |             | 0.70        |
| Proposal |                   |             | 0.557       |
