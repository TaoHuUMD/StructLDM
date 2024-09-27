import torch
import torch.nn as nn
import torch.nn.functional as F

#from Eng.modules.losses.lpips import LPIPS
#from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

from uvm_lib.engine.thutil.networks.loss_vqgan.lpips import LPIPS
from uvm_lib.engine.thutil.networks.discriminator.model import NLayerDiscriminator, weights_init

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss, loss_real, loss_fake


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=0, split="train", img_mask = None):
        
        if cond != 0:
            inputs, _ = torch.split(inputs, [inputs.shape[1] - cond, cond], dim=1)
            reconstructions, cinput_ = torch.split(reconstructions, [reconstructions.shape[1] - cond, cond], dim=1)
            cond_input = cinput_

        if img_mask is not None:
            img_mask_ = img_mask.repeat(1, int(inputs.shape[1] // img_mask.shape[1]), 1, 1)
            pixel_loss = torch.abs(inputs.contiguous() * img_mask_ - reconstructions.contiguous() * img_mask_) * self.pixel_weight
        else:
            pixel_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) * self.pixel_weight

        if self.perceptual_weight > 0:
            #percept_loss = []
            percept_loss = torch.tensor([0.0]).to(inputs.device)
            i = 0
            while i < inputs.shape[1]:
                #percept_loss.append(self.perceptual_loss(inputs.contiguous()[:,i:i+3,...], reconstructions.contiguous()[:,i:i+3,...]))
                percept_loss += (self.perceptual_loss(inputs.contiguous()[:,i:i+3,...], reconstructions.contiguous()[:,i:i+3,...]))[0][0][0]
                i += 3
            percept_loss /= (inputs.shape[1] // 3)
            percept_loss = self.perceptual_weight * percept_loss #torch.mean(percept_loss)
        else:
            percept_loss = torch.tensor([0.0])
            #rec_loss = pixel_loss

        nll_loss = pixel_loss.to(inputs.device) + percept_loss.to(inputs.device)
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None or cond == 0:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond_input), dim=1))
            g_loss = -torch.mean(logits_fake)

            if last_layer is None:
                d_weight = 1.0
            else:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            g_gan_loss = d_weight * disc_factor * g_loss #+ self.codebook_weight * codebook_loss.mean()

            if False:
                log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                    #"{}/quant_loss".format(split): codebook_loss.detach().mean(),
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/p_loss".format(split): p_loss.detach().mean(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),                   
                    }
            return  pixel_loss.mean(), percept_loss.mean(), g_gan_loss, d_weight 
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None or cond == 0:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond_input), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond_input), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss, real_loss, fake_loss = self.disc_loss(logits_real, logits_fake)
            d_loss *= disc_factor

            return d_loss, real_loss, fake_loss

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
