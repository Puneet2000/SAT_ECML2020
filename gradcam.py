import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fn
import torch.nn as nn
from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer
import numpy as np
import torchvision
from torch.autograd import Variable
class GradCAM(object):
    def __init__(self, model_type,layer_name,model):
        self.model_arch = model

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch.feature, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)


    def forward(self, input, class_idx=None, retain_graph=True):
        b, c, h, w = input.size()

        logit,_ = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):

    def __init__(self, model_type,layer_name,model):
        super(GradCAMpp, self).__init__(model_type,layer_name,model)
        self.b = False
        self.model_arch = model

    def forward(self, input, class_idx=None, retain_graph=True):

        b, c, h, w = input.size()

        logit,_ = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze() 
            
        self.model_arch.zero_grad()
        score.backward(retain_graph=True)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.upsample(saliency_map, size=(32, 32), mode='bilinear', align_corners=False)
        # saliency_map = torch.sign(saliency_map)
        
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map


class GuidedBackprop(object):
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.handlers = []
        def backward_hook(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))

    def forward(self, input_image):
        # Forward pass
        input_image.requires_grad = True
        model_output,_ = self.model(input_image)
        target_class = int(torch.argmax(model_output,1))
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output.cuda())
        gradient = input_image.grad.clone()
        # gradient = gradient - gradient.min()
        # gradient = gradient/gradient.max()
        gradient = torch.sign(gradient)
        input_image.grad.zero_()
        return gradient

class GuidedGradCAM(object):
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self,model_type,layer_name,model):
        self.model = model
        self.model_type = model_type
        self.layer_name = layer_name
        self.gradcam =  GradCAMpp(self.model_type,self.layer_name,self.model)
        self.gbp = GuidedBackprop(self.model)
        
    def forward(self, input_image):
        g_mask= self.gradcam(input_image)
        gradient = self.gbp.forward(input_image)
        mask = (gradient+1)*(g_mask+1)
        mask = (mask-2.)/2.
        return mask

def convert_to_grayscale(im_as_arr):

    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

class VanillaGrad(object):

    def __init__(self, pretrained_model):
        self.pretrained_model = pretrained_model

    def forward(self, x, index=None):
        output,_ = self.pretrained_model(x)

        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
        one_hot = torch.sum(one_hot * output)

        one_hot.backward()

        grad = x.grad.data

        return grad


class SmoothGrad(VanillaGrad):

    def __init__(self, pretrained_model, stdev_spread=0.15,
                 n_samples=10):
        super(SmoothGrad, self).__init__(pretrained_model)

        self.stdev_spread = stdev_spread
        self.n_samples = n_samples

    def forward(self, x, index=None):
        x = x.data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(x) - np.min(x))
        total_gradients = np.zeros_like(x)
        for i in range(self.n_samples):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            x_plus_noise = Variable(torch.from_numpy(x_plus_noise).cuda(), requires_grad=True)
            output,_ = self.pretrained_model(x_plus_noise)

            if index is None:
                index = np.argmax(output.data.cpu().numpy())

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
            one_hot = torch.sum(one_hot * output)

            if x_plus_noise.grad is not None:
                x_plus_noise.grad.data.zero_()
            one_hot.backward()

            grad = x_plus_noise.grad.data.cpu().numpy()

            total_gradients += grad
            #if self.visdom:

        avg_gradients = torch.from_numpy(total_gradients/ self.n_samples).cuda()
        avg_gradients = avg_gradients - avg_gradients.min()
        avg_gradients = avg_gradients/avg_gradients.max()
        # avg_gradients = torch.sign(avg_gradients)

        return avg_gradients

class IntegratedGrad(object):

    def __init__(self, pretrained_model):
        super(IntegratedGrad, self).__init__()

        self.model = pretrained_model
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, img, y):
        img = img.squeeze(0)
        steps = 100
        inputs_sq = img.detach().squeeze().cpu().numpy() 
        baseline = 0 * inputs_sq
        scaled_inputs = torch.from_numpy( np.array([baseline + (float(i) / steps) * (inputs_sq - baseline) for i in range(0, steps + 1)]))
        scaled_inputs = Variable(scaled_inputs).cuda()
        scaled_inputs.requires_grad = True
        scores,_ = self.model(scaled_inputs)
        loss = self.lossfn(scores, y.repeat(steps+1).cuda())
        self.model.zero_grad()
        gradient = torch.autograd.grad(loss, scaled_inputs )[0]
        gradient = gradient.detach()
        
        avg_grads = torch.div(gradient[1:] + gradient[:-1] ,2.0)
        avg_grads = torch.mean(avg_grads, dim=0)
        integrated_grad = (torch.from_numpy(inputs_sq) - torch.from_numpy(baseline)) * avg_grads.cpu()
        integrated_grad = integrated_grad - integrated_grad.min()
        integrated_grad =  integrated_grad/integrated_grad.max()
        # integrated_grad = torch.sign(integrated_grad)
        return integrated_grad.unsqueeze(0).cuda()