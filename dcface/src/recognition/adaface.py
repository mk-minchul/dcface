from torch.nn import Module, Parameter
import math
import torch


class AdaFaceV3(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332,  m=0.5, scaler_fn=None, rad_h=0.0, s=64., t_alpha=0.01, cut_gradient=False, head_b=0.5):
        super(AdaFaceV3, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.eps = 1e-3
        self.h = rad_h
        self.s = s

        self.scaler_fn = scaler_fn
        self.cut_gradient = cut_gradient
        self.head_b = head_b

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))

        self.register_buffer('batch_mean', torch.ones(1)*(0))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\ncreating AdaFaceV3 with the following property')
        print('self.scaler_fn', self.scaler_fn)
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)
        print('self.cut_gradient', self.cut_gradient)
        print('self.head_b', self.head_b)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        if self.scaler_fn == 'batchnorm_prob':
            margin_scaler = margin_scaler * self.h # 66% between -0.333 ,0.333 h:0.333
            margin_scaler = torch.clip(margin_scaler, -1, 1)
            margin_scaler = (margin_scaler + 1.0) / 2.0 # 66% between 0.333, 0.666 (0~1)
            # ex: m=0.5, h:0.333
            # range
            #       (66% range)
            #    0  0.333  0.666   1  (scaler)
            #    0  0.167  0.333 0.5  (m*scaler)

        elif self.scaler_fn == 'batchnorm':
            margin_scaler = margin_scaler * self.h # 66% between -0.333 ,0.333 h:0.333
            margin_scaler = torch.clip(margin_scaler, -1, 1)
            # ex: m=0.5, h:0.333
            # range
            #       (66% range)
            #   -1 -0.333  0.333   1  (scaler)
            # -0.5 -0.166  0.166 0.5  (m*scaler)
        elif self.scaler_fn == 'curriculum':
            # curriculum
            target_logit = cosine[torch.arange(0, embbedings.size(0)), label].view(-1, 1)
            with torch.no_grad():
                self.t = target_logit.mean() * self.t_alpha + (1 - self.t_alpha) * self.t
                if target_logit.dtype == torch.float16:
                    t = self.t.half()
                else:
                    t = self.t
                margin_scaler = torch.clip(t, 0, 0.7) / (0.7/1.5) # between 0 and 1.5; 0 in the beginning.
                margin_scaler = 1 - margin_scaler # 1 in the beginning (focus on arcface)
                #      begin    end
                #s      0       1.5
                #1-s    1       -0.5
                #m(1-s) 0.5     -0.25

        else:
            raise ValueError('not a correct scaler')

        # arcface
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        m_arc = m_arc * self.m * margin_scaler
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # cosface
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        m_cos = m_cos * (self.head_b - (self.m * margin_scaler))
        cosine = cosine - m_cos

        if self.cut_gradient:
            cos_theta_yi = cosine[torch.arange(0, embbedings.size(0)), label].view(-1, 1)
            grad_scaler = torch.cos(self.m * margin_scaler) + cos_theta_yi * torch.sin(self.m * margin_scaler) / torch.sqrt(1-cos_theta_yi**2)
            bad_grad = grad_scaler < 0
            scaled_cosine_m = cosine * self.s
            return scaled_cosine_m, bad_grad

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


def l2_norm(input,axis=1):
    norm = torch.clip(torch.norm(input,2,axis,True), 1e-5)
    output = torch.div(input, norm)
    return output
