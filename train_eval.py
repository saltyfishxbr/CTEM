import os
import json

from jinja2.optimizer import optimize
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData,InMemoryDataset
from torch_geometric.loader import DataLoader

from utils import EgoLoss,CTEDistillLoss
from network import CTEGraphPredictor

# EDGE_TYPES = [
#     ('small', 'interacts', 'small'),
#     ('small', 'interacts', 'large'),
#     ('small', 'interacts', 'motor'),
#     ('large', 'interacts', 'small'),
#     ('large', 'interacts', 'motor'),
#     ('motor', 'interacts', 'small'),
# ]
# TYPE_ID2NAME={1:'motor',2:'small',3:'large'}

NODE_TYPES = ['acc', 'small', 'large']
EDGE_TYPES = [
    (src, 'interacts', dst)
    for src in NODE_TYPES
    for dst in NODE_TYPES
]
TYPE_ID2NAME = {1: 'acc', 2: 'small', 3: 'large'}


class NGSIMDataset(Dataset):
    def __init__(self,npz_path,split='train'):
        arr=np.load(npz_path,allow_pickle=True)
        self.ego_hist=arr['ego_hist']
        self.ego_future=arr['ego_future']
        self.ego_type=arr['ego_type']
        self.cand_hist=arr['cand_hist']
        self.cand_mask=arr['cand_mask']
        self.cand_type=arr['cand_type']
        self.topk_te=arr['topk_te']
        self.reverse_te=arr['reverse_te']
        self.cand_phys=arr['cand_phys']
        self.dt=float(arr['dt'][0])
        self.idx_split=arr['train_idx'] if split=='train' else arr['val_idx']

    def __len__(self):
        return len(self.idx_split)

    def __getitem__(self, i):
        idx=int(self.idx_split[i])
        ego_hist=torch.from_numpy(self.ego_hist[idx]).float()#[T,4]
        ego_future=torch.from_numpy(self.ego_future[idx]).float()#[T,2]
        cand_hist=torch.from_numpy(self.cand_hist[idx]).float()#[K,T,4]
        cand_mask=torch.from_numpy(self.cand_mask[idx]).float()#[K,T]
        cand_type=torch.from_numpy(self.cand_type[idx]).long()#[K]
        topk_te=torch.from_numpy(self.topk_te[idx]).float()#[K]
        rev_te=torch.from_numpy(self.reverse_te[idx]).float()#[K]
        cand_phys=torch.from_numpy(self.cand_phys[idx]).float()#[K,T,4]
        ego_type_id=int(self.ego_type[idx])
        ego_type=TYPE_ID2NAME[ego_type_id]

        data=HeteroData()

        # motor_idx=(cand_type==1).nonzero(as_tuple=False).flatten()
        # small_idx=(cand_type==2).nonzero(as_tuple=False).flatten()
        # large_idx=(cand_type==3).nonzero(as_tuple=False).flatten()

        acc_idx = (cand_type == 1).nonzero(as_tuple=False).flatten()  # 自动驾驶
        small_idx = (cand_type == 2).nonzero(as_tuple=False).flatten()
        large_idx = (cand_type == 3).nonzero(as_tuple=False).flatten()

        def pack_nodes(nt,idxs):
            xs,msks=[],[]
            for j in idxs.tolist():
                xs.append(cand_hist[j])#[T,4]
                msks.append(cand_mask[j])#[T]
            ego_local_idx=None
            if nt==ego_type:
                xs.append(ego_hist)
                msks.append(torch.ones_like(cand_mask[0]))
                ego_local_idx=len(xs)-1
            if len(xs)==0:
                X=torch.zeros(0,ego_hist.size(0),ego_hist.size(1))
                M=torch.zeros(0,ego_hist.size(0))
            else:
                X=torch.stack(xs,dim=0)
                M=torch.stack(msks,dim=0)
            return X,M,ego_local_idx

        # data['motor'].x,data['motor'].mask,ego_idx_motor=pack_nodes('motor',motor_idx)
        # data['small'].x, data['small'].mask, ego_idx_small = pack_nodes('small', small_idx)
        # data['large'].x, data['large'].mask, ego_idx_large = pack_nodes('large', large_idx)

        data['acc'].x, data['acc'].mask, ego_idx_acc = pack_nodes('acc', acc_idx)
        data['small'].x, data['small'].mask, ego_idx_small = pack_nodes('small', small_idx)
        data['large'].x, data['large'].mask, ego_idx_large = pack_nodes('large', large_idx)


        # if ego_type=='motor' and ego_idx_motor is not None:
        #     data['motor'].y=ego_future
        #     data['motor'].y_mask=torch.ones(ego_future.size(0))
        #     data['ego_type']='motor'
        #     data['ego_index']=torch.tensor(ego_idx_motor).long()
        # elif ego_type=='small' and ego_idx_small is not None:
        #     data['small'].y=ego_future
        #     data['small'].y_mask=torch.ones(ego_future.size(0))
        #     data['ego_type']='small'
        #     data['ego_index']=torch.tensor(ego_idx_small).long()
        # else:
        #     data['large'].y=ego_future
        #     data['large'].y_mask=torch.ones(ego_future.size(0))
        #     data['ego_type']='large'
        #     data['ego_index']=torch.tensor(ego_idx_large).long()

        if ego_type == 'acc' and ego_idx_acc is not None:
            data['acc'].y = ego_future
            data['acc'].y_mask = torch.ones(ego_future.size(0))
            data['ego_type'] = 'acc'
            data['ego_index'] = torch.tensor(ego_idx_acc).long()
        elif ego_type == 'small' and ego_idx_small is not None:
            data['small'].y = ego_future
            data['small'].y_mask = torch.ones(ego_future.size(0))
            data['ego_type'] = 'small'
            data['ego_index'] = torch.tensor(ego_idx_small).long()
        else:
            data['large'].y = ego_future
            data['large'].y_mask = torch.ones(ego_future.size(0))
            data['ego_type'] = 'large'
            data['ego_index'] = torch.tensor(ego_idx_large).long()

        def phys_mean(j):
            pm=cand_phys[j]
            mm=cand_mask[j]
            if mm.sum()>0:
                m=(pm*mm.unsqueeze(-1)).sum(0)/mm.sum()
            else:
                m=pm.mean(0)
            return m[0].item(),m[1].item(),m[2].item(),m[3].item()

        def phys_last(j):
            pm=cand_phys[j][-1,:]
            return pm[0].item(),pm[1].item(),pm[2].item(),pm[3].item()

        def add_edges(src_type,src_indices):
            if src_indices.numel()==0:
                return
            if data['ego_type']==src_type:
                ego_local=int(data['ego_index'])
            else:
                ego_local=data[data['ego_type']].x.size(0)-1

            E=src_indices.numel()
            src=torch.arange(E,dtype=torch.long)
            edge_index=torch.stack([src,torch.full((E,),ego_local,dtype=torch.long)],dim=0)
            attrs=[]
            for ii,j in enumerate(src_indices.tolist()):
                te_fwd=topk_te[j].item()
                te_rev=rev_te[j].item()
                dist_m,rel_v,ttc_m,same_lane=phys_last(j)
                attrs.append([te_fwd,te_rev,dist_m,rel_v,ttc_m,same_lane])
            edge_attr=torch.tensor(attrs,dtype=torch.float32)
            data[(src_type,'interacts',data['ego_type'])].edge_index=edge_index
            data[(src_type,'interacts',data['ego_type'])].edge_attr=edge_attr

            edge_index_rev = torch.stack([torch.full((E,), ego_local, dtype=torch.long), src], dim=0)
            data[(data['ego_type'], 'interacts', src_type)].edge_index = edge_index_rev
            data[(data['ego_type'], 'interacts', src_type)].edge_attr = edge_attr.clone()

        # add_edges('motor',motor_idx)
        # add_edges('small',small_idx)
        # add_edges('large',large_idx)
        add_edges('acc', acc_idx)
        add_edges('small', small_idx)
        add_edges('large', large_idx)

        return data

def _as_scalar(x):
    import torch
    if isinstance(x, (list, tuple)):
        return _as_scalar(x[0])
    if torch.is_tensor(x):
        if x.ndim == 0 or x.numel() == 1:
            return x.view(-1)[0].item()
        return _as_scalar(x.view(-1)[0])
    return x

def train(model,loader,optimizer,epoch,device,writer,log_interval=10):
    model.train()
    ego_loss_fn=EgoLoss()
    distill_loss_fn=CTEDistillLoss()
    total_loss = 0

    for i,data in enumerate(tqdm(loader)):
        data=data.to(device)
        optimizer.zero_grad()

        pred,_,pred_cte,true_cte=model(data)

        # ego_type=data['ego_type']
        # ego_index=int(data['ego_index'])
        # gt=data[ego_type].y[ego_index].unsqueeze(0)
        # mask=data[ego_type].y_mask[ego_index].unsqueeze(0)
        ego_type = _as_scalar(data['ego_type'])  # 'small' / 'large' / 'motor'
        ego_index = int(_as_scalar(data['ego_index']))  # Python int

        store = data[ego_type]  # 现在是 Hetero 节点存储
        gt = store.y[ego_index].unsqueeze(0)  # [1, Tp, 2]
        mask = store.y_mask[ego_index].unsqueeze(0)  # [1, Tp]

        ego_loss=ego_loss_fn(pred.unsqueeze(0),gt,mask)
        distill_loss=distill_loss_fn(pred_cte,true_cte)
        loss=ego_loss+distill_loss

        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        if i % log_interval == 0:
            writer.add_scalar('train/loss_batch',loss.item(),epoch*len(loader)+i)
            writer.add_scalar('train/ego_loss',ego_loss.item(),epoch*len(loader)+i)
            writer.add_scalar('train/distill_loss',distill_loss.item(),epoch*len(loader)+i)

    writer.add_scalar('train/loss_epoch',total_loss/len(loader),epoch)

@torch.no_grad()
def evaluate(model,loader,epoch,device,writer):
    model.eval()
    ego_loss_fn=EgoLoss()
    distill_loss_fn=CTEDistillLoss()
    total_loss = 0

    for data in loader:
        data=data.to(device)
        pred, _, pred_cte, true_cte = model(data)

        # ego_type = data['ego_type']
        # ego_index = int(data['ego_index'])
        # gt = data[ego_type].y[ego_index].unsqueeze(0)
        # mask = data[ego_type].y_mask[ego_index].unsqueeze(0)
        ego_type = _as_scalar(data['ego_type'])  # 'small' / 'large' / 'motor'
        ego_index = int(_as_scalar(data['ego_index']))  # Python int

        store = data[ego_type]  # 现在是 Hetero 节点存储
        gt = store.y[ego_index].unsqueeze(0)  # [1, Tp, 2]
        mask = store.y_mask[ego_index].unsqueeze(0)  # [1, Tp]

        ego_loss = ego_loss_fn(pred.unsqueeze(0), gt, mask)
        distill_loss = distill_loss_fn(pred_cte, true_cte)
        loss =ego_loss + distill_loss

        total_loss+=loss.item()

    avg_loss=total_loss/len(loader)
    writer.add_scalar('val/loss',avg_loss,epoch)
    return avg_loss

def main(npz_path,log_dir='./logs/TGSIM_L2/cv/50',ckpt_dir='./ckpt/TGSIM_L2/cv/50',lr=1e-3,epochs=30,batch_size=1):
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(ckpt_dir,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds=NGSIMDataset(npz_path,split='train')
    val_ds=NGSIMDataset(npz_path,split='val')
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False)

    model=CTEGraphPredictor().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5)
    writer=SummaryWriter(log_dir=log_dir)

    best=float('inf')
    for epoch in range(1,epochs+1):
        print(f'===>{epoch}/{epochs}')
        train(model,train_loader,optimizer, epoch, device, writer)
        val_loss=evaluate(model,val_loader,epoch,device,writer)
        scheduler.step(val_loss)
        if val_loss<best:
            best=val_loss
            torch.save(model.state_dict(),os.path.join(ckpt_dir,'best_model.pth'))
        torch.save(model.state_dict(),os.path.join(ckpt_dir,f'epoch_{epoch}_{val_loss:.4f}.pth'))

    writer.close()
    print('Training completed. Best val loss:',best)

if __name__=='__main__':
    main(npz_path='./ProcessedData/TGSIM_L2/_topk_dataset.npz')