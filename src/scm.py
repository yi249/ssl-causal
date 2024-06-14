from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.distributions as dist
from torch.distributions.kl import kl_divergence as kl
import torch.nn.functional as F
import pyro.distributions.transforms as T
from torch import Tensor, nn

from hps import Hparams
from layers import (  # fmt: skip
    CNN,
    ConditionalAffineTransform,
    ConditionalGumbelMax,
    ConditionalTransformedDistributionGumbelMax,
)
from pyro.nn import DenseNN

def bernoulli_kl(p,q):
    kl = p*torch.log(p/q) + (1-p)* torch.log((1-p)/(1-q))
    return torch.sum(kl)

def bernoulli_log_prob(p,x):
    ll = torch.log(p)*x + torch.log(1-p)*(1-x)
    return torch.sum(ll)

class SSL_Classifier(nn.Module):
    def __init__(self, args: Hparams):
        super().__init__()
        self.hps = args.hps
        self.device = args.device
        self.input_res = args.input_res
        self.dataset = args.hps
        self.resolutions = []

        ## CMMNIST
        if self.hps == "cmmnist":
            self.digit_logits = nn.Parameter(torch.zeros(1, 10))  # uniform prior
            for k in ["t", "i", "fg", "bg"]:  # thickness, intensity, standard Gaussian
                self.register_buffer(f"{k}_base_loc", torch.zeros(1))
                self.register_buffer(f"{k}_base_scale", torch.ones(1))

            # constraint, assumes data is [-1,1] normalized
            normalize_transform = T.ComposeTransform(
                [T.SigmoidTransform(), T.AffineTransform(loc=-1, scale=2)]
            )

            self.fg_net = DenseNN(10, args.flow_widths, [3, 3], nonlinearity=nn.GELU())
            self.fg_context_nn = ConditionalAffineTransform(
                context_nn=self.fg_net, event_dim=0
            )
            self.fg_flow = [self.fg_context_nn]

            # background colour flow
            self.bg_net = DenseNN(10, args.flow_widths, [3, 3], nonlinearity=nn.GELU())
            self.bg_context_nn = ConditionalAffineTransform(
                context_nn=self.bg_net, event_dim=0
            )
            self.bg_flow = [self.bg_context_nn]

            # thickness flow
            self.thickness_module = T.ComposeTransformModule(
                [T.Spline(1, count_bins=4, order="linear")]
            )
            self.thickness_flow = T.ComposeTransform(
                [self.thickness_module, normalize_transform]
            )

            # intensity (conditional) flow: thickness -> intensity
            self.intensity_net = DenseNN(1, args.flow_widths, [1, 1], nonlinearity=nn.GELU())
            self.context_nn = ConditionalAffineTransform(
                context_nn=self.intensity_net, event_dim=0
            )
            self.intensity_flow = [self.context_nn, normalize_transform]

            # anticausal predictors
            input_shape = (args.input_channels, args.input_res, args.input_res)
            # q(t | x, i) = Normal(mu(x, i), sigma(x, i)), 2 outputs: loc & scale
            self.encoder_t = CNN(input_shape, num_outputs=2, context_dim=1, width=8)
            # q(i | x) = Normal(mu(x), sigma(x))
            self.encoder_i = CNN(input_shape, num_outputs=2, width=8)
            # q(fg | x) = Normal(mu(x), sigma(x))
            self.encoder_fg = CNN(input_shape, num_outputs=6, width=8)
            # q(bg | x) = Normal(mu(x), sigma(x))
            self.encoder_bg = CNN(input_shape, num_outputs=6, width=8)
            # q(y | x, fg, bg) = Categorical(pi(x))
            self.encoder_d = CNN(input_shape, num_outputs=10, context_dim=6, width=8)

            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x)+0.0001
            )

            self.std = lambda x: F.sigmoid(x)/20
            self.stdi = lambda x: F.sigmoid(x)/100

        ## MIMIC
        if self.hps == "mimic192":
            # Discrete variables that are not root nodes
            self.discrete_variables = {"finding": "binary"}
            # define base distributions
            for k in ["a"]:
                self.register_buffer(f"{k}_base_loc", torch.zeros(1))
                self.register_buffer(f"{k}_base_scale", torch.ones(1))

            # age spline flow
            self.age_flow_components = T.ComposeTransformModule([T.Spline(1)])
            self.age_flow = T.ComposeTransform(
                [
                    self.age_flow_components,
                ]
            )
            # Finding (conditional) via MLP, a -> f
            self.finding_net = DenseNN(1, [8, 16], param_dims=[1], nonlinearity=nn.Sigmoid())
            self.finding_transform_GumbelMax = ConditionalGumbelMax(
                context_nn=self.finding_net, event_dim=0
            )
            # log space for sex and race
            self.sex_prob = nn.Parameter(0.4624 * torch.ones(1))
            self.race_logits = nn.Parameter(np.log(1 / 3) * torch.ones(1, 3))


            from resnet import CustomBlock, ResNet, ResNet18

            shared_model = ResNet(
                CustomBlock,
                layers=[2, 2, 2, 2],
                widths=[64, 128, 256, 512],
                norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c),
            )
            # shared_model = torchvision.models.resnet18(weights=None)
            shared_model.conv1 = nn.Conv2d(
                args.input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            kwargs = {
                "in_shape": (args.input_channels, *(args.input_res,) * 2),
                "base_model": shared_model,
            }
            # q(s | x) ~ Bernoulli(f(x))
            self.encoder_s = ResNet18(num_outputs=1, **kwargs)
            # q(r | x) ~ OneHotCategorical(f(x))
            self.encoder_r = ResNet18(num_outputs=3, **kwargs)
            # q(f | x) ~ Bernoulli(f(x))
            self.encoder_f = ResNet18(num_outputs=1, **kwargs)
            # q(a | x, f) ~ Normal(mu(x), sigma(x))
            self.encoder_a = ResNet18(num_outputs=2, context_dim=1, **kwargs)
            self.f = (
                lambda x: args.std_fixed * torch.ones_like(x)
                if args.std_fixed > 0
                else F.softplus(x) + 0.0001
            )

            self.std = lambda x: F.sigmoid(x) / 20
            self.stdi = lambda x: F.sigmoid(x) / 100

    def expand(self, variable):
        expanded = variable[..., None, None].repeat(1, 1, *(self.input_res,) * 2)
        return expanded.to(self.device)
    def losses(self, x: Dict[int, Tensor], use_scm=False, truth=None, weight=0):
        def replace_nans(original: Tensor, replacement: Tensor):
            if original.size() != replacement.size():
                raise ValueError("Input tensors must have the same size")
            nan_mask = torch.isnan(original)
            result_tensor = torch.where(nan_mask, replacement, original)
            return result_tensor

        def condition_nans(nans: Tensor, condition: Tensor):
            if nans.size()[0] != condition.size()[0]:
                raise ValueError("Input tensors must have the same length")
            shape = condition.shape[1]
            new_nans = torch.swapaxes(nans[:, 0].repeat(shape, 1), 0, 1)
            result = condition[new_nans]
            result = torch.reshape(result, (-1,shape))
            return result

        elbo = 0.0
        mse = nn.MSELoss()
        l1 = nn.L1Loss()
        ce = nn.CrossEntropyLoss()
        bce = nn.BCELoss()
        sup_loss = torch.tensor(0.0).cuda()

        for k,v in truth.items():
            if v.dim() == 4:
                truth[k] = v[...,0,0]

        new_x = x#self.forward(x).to(self.device)
        batch_size = new_x.shape[0]
        predictions = {}

        unif = dist.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        norm = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        mvn = dist.MultivariateNormal(torch.zeros(3),torch.eye(3))
        softmax = nn.Softmax(dim=1)
        sigmoid = nn.Sigmoid()


        if self.dataset == "cmmnist":

            i_nans = torch.isnan(truth['intensity'])
            t_nans = torch.isnan(truth['thickness'])
            fg_nans = torch.isnan(truth['fgcol'])
            bg_nans = torch.isnan(truth['bgcol'])
            d_nans = torch.isnan(truth['digit'])

            not_nans = ~(i_nans[:, 0] | t_nans[:, 0] | fg_nans[:, 0] | bg_nans[:, 0] | d_nans[:, 0])

            ## Encoder
            # q(i | x)
            i_loc, i_logscale = self.encoder_i(new_x).chunk(2, dim=-1)
            if not torch.all(i_nans == False): #deal with case when entire batch is supervised for this var
                qi_x = dist.Normal(torch.tanh(i_loc)[i_nans], self.std(i_logscale)[i_nans])
            samp = norm.sample().to(self.device)
            predictions['intensity'] = self.std(i_logscale)*samp + torch.tanh(i_loc)
            truth['intensity'] = replace_nans(truth['intensity'], predictions['intensity'])
            supervised_preds = condition_nans(~i_nans, torch.tanh(i_loc)) #predictions['intensity'])
            supervised_truth = condition_nans(~i_nans, truth['intensity'])
            i_loss = l1(supervised_preds, supervised_truth)
            if i_loss == i_loss:
                sup_loss += i_loss

            # q(t | x, i)
            t_loc, t_logscale = self.encoder_t(new_x, y=truth["intensity"]).chunk(2, dim=-1)
            if not torch.all(t_nans == False):
                qt_x = dist.Normal(torch.tanh(t_loc)[t_nans], self.std(t_logscale)[t_nans])
            samp = norm.sample().to(self.device)
            predictions['thickness'] = self.std(t_logscale)*samp + torch.tanh(t_loc)
            truth['thickness'] = replace_nans(truth['thickness'], predictions['thickness'])
            supervised_preds = condition_nans(~t_nans, torch.tanh(t_loc))#predictions['thickness'])
            supervised_truth = condition_nans(~t_nans, truth['thickness'])
            t_loss = l1(supervised_preds, supervised_truth)
            if t_loss == t_loss:
                sup_loss += t_loss

            # q(fg | x)
            fg_loc, fg_logscale = self.encoder_fg(new_x).chunk(2, dim=-1)
            if not torch.all(fg_nans == False):
                qfg_x = dist.MultivariateNormal(condition_nans(fg_nans, torch.tanh(fg_loc)),
                                    torch.diag_embed(condition_nans(fg_nans, self.std(fg_logscale))))
            samp = mvn.sample().to(self.device)
            predictions['fgcol'] = self.std(fg_logscale)*samp + torch.tanh(fg_loc)
            truth['fgcol'] = replace_nans(truth['fgcol'], predictions['fgcol'])
            supervised_preds = condition_nans(~fg_nans, torch.tanh(fg_loc))#predictions['fgcol'])
            supervised_truth = condition_nans(~fg_nans, truth['fgcol'])
            fg_loss = l1(supervised_preds, supervised_truth)
            if fg_loss == fg_loss:
                sup_loss += fg_loss

            # q(bg | x)
            bg_loc, bg_logscale = self.encoder_bg(new_x).chunk(2, dim=-1)
            if not torch.all(bg_nans == False):
                qbg_x = dist.MultivariateNormal(condition_nans(bg_nans, torch.tanh(bg_loc)),
                                    torch.diag_embed(condition_nans(bg_nans, self.std(bg_logscale))))
            samp = mvn.sample().to(self.device)
            predictions['bgcol'] = self.std(bg_logscale)*samp + torch.tanh(bg_loc)
            truth['bgcol'] = replace_nans(truth['bgcol'], predictions['bgcol'])
            supervised_preds = condition_nans(~bg_nans, torch.tanh(bg_loc))#predictions['bgcol'])
            supervised_truth = condition_nans(~bg_nans, truth['bgcol'])
            bg_loss = l1(supervised_preds, supervised_truth)
            if bg_loss == bg_loss:
                sup_loss += bg_loss

            # q(d | x, fg, bg)
            d_prob = F.softmax(self.encoder_d(new_x, y=torch.cat((truth["fgcol"], truth["bgcol"]), 1)), dim=-1)
            if not torch.all(d_nans == False):
                qd_x = dist.OneHotCategorical(probs=condition_nans(d_nans,d_prob))  # .to_event(1)
            uniform = torch.rand(d_prob.shape).to(self.device)
            gumbels = -torch.log(-torch.log(uniform)) + torch.log(d_prob)
            predictions['digit'] = softmax(gumbels)
            weights = 1+torch.sum(predictions['digit']*torch.log(predictions['digit'])/np.log(10), dim=1)
            truth['digit'] = replace_nans(truth['digit'], predictions['digit'])
            supervised_preds = condition_nans(~d_nans, predictions['digit'])
            supervised_truth = condition_nans(~d_nans, truth['digit'])
            d_loss = ce(supervised_preds, torch.argmax(supervised_truth, dim=1))
            if d_loss == d_loss:
                sup_loss += d_loss
            d_acc = torch.mean([torch.argmax(supervised_preds, dim=1) == torch.argmax(supervised_truth, dim=1)][0].to(torch.float32))

            ## Decoder


            # p(d)
            pd = dist.OneHotCategorical(
                probs=F.softmax(self.digit_logits, dim=-1)
            )
            log_prob = pd.log_prob(F.one_hot(torch.argmax(truth['digit'], dim=1),num_classes=10))
            seen_mean = log_prob[~d_nans[:,0]].mean()
            if seen_mean == seen_mean:
                elbo += seen_mean
            if not torch.all(d_nans == False):
                elbo -= kl(qd_x, pd).mean()

            # p(fg | d)
            if not torch.all(fg_nans == False): #case when everything seen so no latents of this var
                fg_loc_latent, fg_logscale_latent = self.fg_net(condition_nans(fg_nans, truth['digit']))
                pfg_d_latent = dist.MultivariateNormal(torch.tanh(fg_loc_latent), torch.diag_embed(self.std(fg_logscale_latent)))
                elbo -= kl(qfg_x, pfg_d_latent).mean()
            if not torch.all(fg_nans == True): #case when nothing seen so no seen of this var
                fg_loc_seen, fg_logscale_seen = self.fg_net(condition_nans(~fg_nans, truth['digit']))
                pfg_d_seen = dist.MultivariateNormal(torch.tanh(fg_loc_seen), torch.diag_embed(self.std(fg_logscale_seen)))
                log_prob = pfg_d_seen.log_prob(condition_nans(~fg_nans, truth['fgcol']))
                elbo += log_prob.mean()


            # p(bg | d)
            if not torch.all(bg_nans == False):
                bg_loc_latent, bg_logscale_latent = self.bg_net(condition_nans(bg_nans, truth['digit']))
                pbg_d_latent = dist.MultivariateNormal(torch.tanh(bg_loc_latent),torch.diag_embed(self.std(bg_logscale_latent)))
                elbo -= kl(qbg_x, pbg_d_latent).mean()
            if not torch.all(bg_nans == True):
                bg_loc_seen, bg_logscale_seen = self.bg_net(condition_nans(~bg_nans, truth['digit']))
                pbg_d_seen = dist.MultivariateNormal(torch.tanh(bg_loc_seen), torch.diag_embed(self.std(bg_logscale_seen)))
                log_prob = pbg_d_seen.log_prob(condition_nans(~bg_nans, truth['bgcol']))
                elbo += log_prob.mean()

            # p(t)
            pt = dist.Normal(self.t_base_loc, self.t_base_scale/20)
            log_prob = pt.log_prob(truth['thickness'])
            seen_mean = log_prob[~t_nans[:,0]].mean()
            if seen_mean == seen_mean:
                elbo += seen_mean
            if not torch.all(t_nans == False):
                elbo -= kl(qt_x, pt).mean()

            # p(i | t)
            if not torch.all(i_nans == False):
                i_loc_latent, i_logscale_latent = self.intensity_net(condition_nans(i_nans, truth['thickness']))
                pi_t_latent = dist.Normal(torch.tanh(i_loc_latent), self.std(i_logscale_latent))
                elbo -= kl(qi_x, pi_t_latent).mean()
            if not torch.all(i_nans == True):
                i_loc_seen, i_logscale_seen = self.intensity_net(condition_nans(~i_nans, truth['thickness']))
                pi_t_seen = dist.Normal(torch.tanh(i_loc_seen), self.std(i_logscale_seen))
                log_prob = pi_t_seen.log_prob(condition_nans(~i_nans, truth['intensity']))
                elbo += log_prob.mean()

            accs = [fg_loss, bg_loss, t_loss, i_loss, d_acc]

        if self.dataset == "mimic192":

            f_nans = torch.isnan(truth['finding'])
            a_nans = torch.isnan(truth['age'])
            s_nans = torch.isnan(truth['sex'])
            r_nans = torch.isnan(truth['race'])

            not_nans = ~(a_nans[:, 0] | f_nans[:, 0] | s_nans[:, 0] | r_nans[:, 0])

            #Encoder

            # q(f|x)
            f_prob = F.sigmoid(self.encoder_f(new_x))
            if not torch.all(f_nans == False):
                qf_x = condition_nans(f_nans, f_prob)
            predictions['finding'] = f_prob #softmax(gumbels)
            weights = 1+torch.sum(predictions['finding']*torch.log(predictions['finding'])/np.log(2), dim=1)
            truth['finding'] = replace_nans(truth['finding'], predictions['finding'])
            supervised_preds = condition_nans(~f_nans, predictions['finding'])
            supervised_truth = condition_nans(~f_nans, truth['finding'])
            f_loss = bce(torch.squeeze(supervised_preds), torch.squeeze(supervised_truth))
            if f_loss == f_loss:
                sup_loss += f_loss
            f_acc = torch.mean(torch.round(supervised_preds.clone().detach()) == supervised_truth.clone().detach(), dtype=torch.float32)

            # q(a | x, f)
            a_loc, a_logscale = self.encoder_a(new_x, y=truth["finding"]).chunk(2, dim=-1)
            if not torch.all(a_nans == False):
                qa_x = dist.Normal(self.f(a_loc)[a_nans], self.f(a_logscale)[a_nans])
            samp = norm.sample().to(self.device)
            predictions['age'] = samp * self.std(a_logscale) + torch.tanh(a_logscale)
            truth['age'] = replace_nans(truth['age'], predictions['age'])
            supervised_preds = condition_nans(~a_nans, predictions['age'])
            supervised_truth = condition_nans(~a_nans, truth['age'])
            a_loss = l1(supervised_preds, supervised_truth)
            if a_loss == a_loss:
                sup_loss += a_loss

            # q(s|x)
            s_prob = F.sigmoid(self.encoder_s(new_x))
            if not torch.all(s_nans == False):
                qs_x = condition_nans(s_nans, s_prob)
            predictions['sex'] = s_prob #softmax(gumbels)
            weights *= 1+torch.sum(predictions['sex']*torch.log(predictions['sex'])/np.log(2), dim=1)
            truth['sex'] = replace_nans(truth['sex'], predictions['sex'])
            supervised_preds = condition_nans(~s_nans, predictions['sex'])
            supervised_truth = condition_nans(~s_nans, truth['sex'])
            s_loss = bce(torch.squeeze(supervised_preds), torch.squeeze(supervised_truth))
            if s_loss == s_loss:
                sup_loss += s_loss
            s_acc = torch.mean(torch.round(supervised_preds.clone().detach()) == supervised_truth.clone().detach(), dtype=torch.float32)


            # q(r|x)
            r_prob = F.softmax(self.encoder_r(new_x), dim=-1)
            if not torch.all(r_nans == False):
                qr_x = dist.OneHotCategorical(probs=condition_nans(r_nans, r_prob))  # .to_event(1)
            uniform = torch.rand(r_prob.shape).to(self.device)
            gumbels = -torch.log(-torch.log(uniform)) + torch.log(r_prob)
            predictions['race'] = softmax(gumbels)
            weights *= 1+torch.sum(predictions['race']*torch.log(predictions['race'])/np.log(3), dim=1)
            truth['race'] = replace_nans(truth['race'], predictions['race'])
            supervised_preds = condition_nans(~r_nans, predictions['race'])
            supervised_truth = condition_nans(~r_nans, truth['race'])
            r_loss = ce(supervised_preds, torch.argmax(supervised_truth, dim=1))
            if r_loss == r_loss:
                sup_loss += r_loss
            r_acc = torch.mean(
                [torch.argmax(supervised_preds, dim=1) == torch.argmax(supervised_truth, dim=1)][0].to(torch.float32))


            #Decoder

            # p(a), age flow
            pa = dist.Normal(self.a_base_loc, self.a_base_scale)
            log_prob = pa.log_prob(truth['age'])
            seen_mean = log_prob[~a_nans[:, 0]].sum()
            if seen_mean == seen_mean:
                elbo += seen_mean
            if not torch.all(a_nans == False):
                elbo -= kl(qa_x, pa).sum()


            # p(f | a), finding as OneHotCategorical conditioned on age
            if not torch.all(f_nans == False):
                f_prob_latent = self.finding_net(condition_nans(f_nans, truth['age']))
                elbo -= bernoulli_kl(sigmoid(qf_x), sigmoid(f_prob_latent))
            if not torch.all(f_nans == True):
                f_prob_seen = self.finding_net(condition_nans(~f_nans, truth['age']))
                log_prob = bernoulli_log_prob(sigmoid(f_prob_seen), condition_nans(~f_nans, truth['finding']))
                elbo += log_prob


            # p(s), sex dist
            ps = torch.tensor(0.424)
            seen_mean = bernoulli_log_prob(ps, condition_nans(~s_nans, truth['sex']))
            if seen_mean == seen_mean:
                elbo += seen_mean
            if not torch.all(s_nans == False):
                elbo -= bernoulli_kl(sigmoid(qs_x), ps)


            # p(r), race dist
            pr = dist.OneHotCategorical(logits=self.race_logits)  # .to_event(1)
            input = F.one_hot(torch.argmax(truth['race'], dim=-1), num_classes=3)
            log_prob = pr.log_prob(input)
            seen_mean = log_prob[~r_nans[:, 0]].sum()
            if seen_mean == seen_mean:
                elbo += seen_mean
            if not torch.all(r_nans == False):
                elbo -= kl(qr_x, pr).sum()

            accs = [f_acc, a_loss, s_acc, r_acc]

        pa_pred = torch.cat([self.expand(var) for var in truth.values()], dim=1)

        if use_scm == False:
            pa_pred = pa_pred[not_nans, ...]
            new_x = new_x[not_nans, ...]
            weights = weights[not_nans]


        return new_x, pa_pred, -elbo, sup_loss, accs, not_nans, weights

    def counterfactuals(self, x: Dict[int, Tensor], parents: Dict[str, Tensor], cf_pa: Dict[str, Tensor]):

        cf = cf_pa.copy()
        pa = parents.copy()

        if self.dataset == "cmmnist":
            if pa['fgcol'] is None:
                pa['fgcol'],_ = torch.tanh(self.encoder_fg(x).chunk(2, dim=-1))
            if pa['bgcol'] is None:
                pa['bgcol'], _ = torch.tanh(self.encoder_bg(x).chunk(2, dim=-1))
            if pa['digit'] is None:
                pa['digit'] = torch.argmax(self.encoder_d(x, y=torch.cat((pa["fgcol"], pa["bgcol"]), 1)), dim=-1)

            if pa['intensity'] is None:
                pa['intensity'],_ = torch.tanh(self.encoder_i(x).chunk(2, dim=-1))
            if pa['thickness'] is None:
                pa['thickness'],_ = torch.tanh(self.encoder_t(x,y=pa['intensity']).chunk(2, dim=-1))

            if cf['digit'] is not None:
                if cf['fgcol'] is None:
                    fg_loc_p, fg_logscale_p = self.fg_net(pa['digit'])
                    pa_params = (torch.tanh(fg_loc_p), self.f(fg_logscale_p))
                    fg_cf_loc_p, fg_cf_logscale_p = self.fg_net(cf['digit'])
                    cf_params = (torch.tanh(fg_cf_loc_p), self.f(fg_cf_logscale_p))
                    cf['fgcol'] = torch.clamp((pa['fgcol'] - pa_params[0]) * cf_params[1] / pa_params[1] + cf_params[0],-1.1)

                if cf['bgcol'] is None:
                    bg_loc_p, bg_logscale_p = self.bg_net(pa['digit'])
                    pa_params = (torch.tanh(bg_loc_p), self.f(bg_logscale_p))
                    bg_cf_loc_p, bg_cf_logscale_p = self.bg_net(cf['digit'])
                    cf_params = (torch.tanh(bg_cf_loc_p), self.f(bg_cf_logscale_p))
                    cf['bgcol'] = torch.clamp((pa['bgcol'] - pa_params[0]) * cf_params[1] / pa_params[1] + cf_params[0],-1,1)

            if cf['thickness'] is not None:
                if cf["intensity"] is None:
                    i_loc_p, i_logscale_p = self.intensity_net(pa['thickness'])
                    pa_params = (torch.tanh(i_loc_p), self.f(i_logscale_p))
                    i_cf_loc_p, i_cf_logscale_p = self.intensity_net(cf['thickness'])
                    cf_params = (torch.tanh(i_cf_loc_p), self.f(i_cf_logscale_p))
                    cf['intensity'] = torch.clamp((pa['intensity'] - pa_params[0]) * cf_params[1] / pa_params[1] + cf_params[0],-1,1)

        if self.dataset == "mimic192":
            if pa['race'] is None:
                pa['race'] = torch.argmax(self.encoder_r(x), dim=-1)
            if pa['sex'] is None:
                pa['sex'] = torch.sigmoid(self.encoder_s(x))
            if pa['finding'] is None:
                pa['finding'] = torch.sigmoid(self.encoder_f(x))
            if pa['age'] is None:
                pa['age'], _ = self.encoder_a(x, y=pa["finding"]).chunk(2, dim=-1)

            if cf['age'] is not None:
                if cf['finding'] is None:
                    f_loc_p, f_logscale_p = self.f_net(pa['age'])
                    pa_params = (torch.sigmoid(f_loc_p), self.f(fg_logscale_p))
                    f_cf_loc_p, f_cf_logscale_p = self.f_net(cf['age'])
                    cf_params = (torch.sigmoid(f_cf_loc_p), self.f(f_cf_logscale_p))
                    cf['finding'] = (pa['finding'] - pa_params[0]) * cf_params[1] / pa_params[1] + cf_params[0]


        for k, v in pa.items():
            if cf[k] is None:
                cf[k] = v
            cf[k] = self.expand(cf[k])
            pa[k] = self.expand(pa[k])


        return pa, cf
