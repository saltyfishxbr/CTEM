import torch
import torch.nn as nn

class ConstantVelocityRollout(nn.Module):
    def __init__(self,dt=0.1,pred_horizon=50):
        super().__init__()
        self.dt=dt
        self.pred_horizon=pred_horizon

    def forward(self,last_pos,last_vel):
        steps=torch.arange(1,self.pred_horizon+1,device=last_pos.device).view(1,-1,1)
        pred=last_pos.unsqueeze(1)+steps*self.dt*last_vel.unsqueeze(1)#[B,T,2]
        return pred

class IDM_MOBIL_Rollout:
    def __init__(self,dt=0.1,default_parameter=None):
        self.dt=dt
        if default_parameter is None:
            default_parameter={
                'v0':30.0,
                'T':1.5,
                'a':1.0,
                'b':1.5,
                's0':2.0
            }
        self.default_parameter=default_parameter

    def compute_acc(self,v,delta_v,s,params):
        s_star=params['s0']+v*params['T']+(v*delta_v)/(2*(params['a']*params['b'])**0.5+1e-6)
        acc=params['a']*(1-(v/params['v0'])**4-(s_star/(s+1e-6))**2)
        return acc

    def forward(self,ego,lead,params=None):
        if params is None:
            params=self.default_parameter
        v=ego['v']
        s=lead['pos']-ego['pos']
        delta_v=v-lead['v']
        acc=self.compute_acc(v,delta_v,s,params)
        v_new=v+acc*self.dt
        pos_new=ego['pos']+v_new*self.dt
        return pos_new,v_new

class RNNRollout(nn.Module):
    def __init__(self,input_dim=4,hidden_dim=64):
        super().__init__()
        self.rnn=nn.GRU(input_dim,hidden_dim,batch_first=True)
        self.out=nn.Linear(hidden_dim,2)

    def forward(self,past_traj):
        out,_=self.rnn(past_traj)
        delta=self.out(out[:,-1:,:])
        return past_traj[:,-1:,:]+delta

class ResidualVelocityRollout(nn.Module):
    def __init__(self, dt=0.1, pred_horizon=50, input_dim=4, hidden_dim=64):
        super().__init__()
        self.dt = dt
        self.pred_horizon = pred_horizon
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, 2)

    def forward(self, hist_pos, hist_vel):
        inp = torch.cat([hist_pos, hist_vel], dim=-1)   # [B,T,4]
        out, _ = self.encoder(inp)
        delta_v = self.regressor(out[:, -1])            # [B,2]
        v_last = hist_vel[:, -1]
        v_new  = v_last + delta_v
        pos_new = hist_pos[:, -1] + v_new * self.dt
        return pos_new, v_new
