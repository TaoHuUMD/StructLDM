from termcolor import colored
import torch

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import random

import Engine.th_utils.grid_sample_fix as grid_sample_fix

#divide input list into num parts
def divide_p(input_list, num, pid_ = -1):
    leng = len(input_list)
    if leng % num == 0:
        sub_list = int(leng / num)
    else:    
        sub_list = leng / num + 0.5
        sub_list = int(sub_list)

    out = []
    for pid in np.arange(num):
        start = sub_list * pid
        end = sub_list * pid + sub_list
        if pid == num - 1:
            end += 1000

        out.append((input_list[start: end], pid))

    return out



def nearest_k_idx(tgt, src, k=3):
   
    r_tgt = torch.sum(tgt * tgt, dim=2, keepdim=True)  # (1,N,x)
    r_src = torch.sum(src * src, dim=2, keepdim=True)  # (B,M,x)

    mul = torch.matmul(src, tgt.permute(0,2,1))         # (B,M,N)
    dist = r_src - 2 * mul + r_tgt.permute(0,2,1)       # (B,M,N)

    knn = dist.topk(k, dim=2, largest=False)

    #indices, (B, M, K)
    return knn.indices

def max_min(v2d):
    for i in range(v2d.shape[1]):
        print(v2d[:,i].max(),v2d[:,i].min(), v2d[:,i].max() - v2d[:,i].min())

def get_random_idx_range(i, min_, max_, range_ = -1):
    begin = min_
    end = max_
    if range_ != -1:
        begin = max(min_, i - range_)
        end = min(max_, i + range_)
    new_index = random.randint(begin, end)

    if new_index == i: 
        return get_random_idx_range(i, min_, max_, range_)
    
    return new_index

def index_2d_img(feat, uv, size=None):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    if size != None:
        uv = (uv - size / 2) / (size / 2)
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in grid_sample_fix.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = grid_sample_fix.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]


def index_2d_UVMap(feat, uv, size=None, padding_mode="zeros"):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, H0, W0] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, H0, W0] image features '''
    
    if feat.ndimension()<4:
        feat = feat[None,...]
        
    #if feat.shape[1] > feat.shape[3]:
    if feat.shape[2] != feat.shape[3]:
        feat = feat.permute(0,3,1,2)
    
    if uv.ndimension() < 4:
        uv = uv.unsqueeze(0)
    
    nbatch = uv.shape[0]
    if uv.shape[-1] == 2:
        normal_flow_b = uv
    else:
        normal_flow_b = uv.permute(0, 2, 3, 1) #TO [B, H0, W0, 2]
          
    flowzero = torch.ones(nbatch, normal_flow_b.shape[1], normal_flow_b.shape[2], 2).float().cuda() * (-5)

    masks = torch.unsqueeze(torch.any(normal_flow_b > 0, 3), -1)

    flow = torch.where(masks, normal_flow_b, flowzero)
    flow = flow * 2 - 1
    
    out_t = grid_sample_fix.grid_sample(feat, flow, mode='nearest', padding_mode=padding_mode, align_corners=True)
      
    return  out_t
    

def index_2D_feature_List(feat, coord_list, size=None):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    if feat.ndimension()<4:
        feat = feat[None,...]
        
    #if feat.shape[1] > feat.shape[3]:
    if feat.shape[2] != feat.shape[-1]:
        feat = feat.permute(0,3,1,2)
    
    if coord_list.ndimension() < 3:
        coord_list = coord_list.unsqueeze(0)

    if not (coord_list.shape[2] == 2):
        coord_list = coord_list.transpose(1, 2)  # [B, N, 2]
        
    coord_list = coord_list.unsqueeze(2)  # [B, N, 1, 2]
    
    #uv_ = uv * 2 - 1
    samples = grid_sample_fix.grid_sample(feat, coord_list, mode='nearest', align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]



def index_2d_UVList(feat, uv, size=None):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    if feat.ndimension()<4:
        feat = feat[None,...]
        
    #if feat.shape[1] > feat.shape[3]:
    if feat.shape[2] != feat.shape[-1]:
        feat = feat.permute(0,3,1,2)
    
    if uv.ndimension() < 3:
        uv = uv.unsqueeze(0)

    if not (uv.shape[2] == 2):
        uv = uv.transpose(1, 2)  # [B, N, 2]
        
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    
    uv_ = uv * 2 - 1
    
    if size != None:
        uv = (uv - size / 2) / (size / 2)
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in grid_sample_fix.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = grid_sample_fix.grid_sample(feat, uv_, mode='nearest', align_corners=True)  # [B, C, N, 1]
    return samples[:, :, :, 0]  # [B, C, N]

def index_3d_volume(feat, uv, size=None, padding_mode="zeros"):#"nearest"
    #already -1, 1
    '''
    :param feat: [B, C, D, H, W] image features
    :param uv: [B, 3, N] duv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''

    if uv.shape[1] ==3 and uv.shape[2]!=3:
        #uv = uv.permute(0,2,1)
        uv = uv.transpose(1, 2)  # [B, N, 3]

    
    uv = uv.unsqueeze(2).unsqueeze(2)  # [B, N, 1, 1, 3]
    
    if size != None:
        uv = (uv - size / 2) / (size / 2)
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in grid_sample_fix.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    
 
    samples = grid_sample_fix.grid_sample(feat, uv, mode="bilinear",
                                    padding_mode = padding_mode, align_corners=True)  # [B, C, N, 1, 1]
    return samples[:, :, :, 0, 0]  # [B, C, N]

def index_2d_dimension(input, index):
    # input: B, X, Y, D
    # index: B, N, 2 ,  2 is (x1, y1)
    # OUTPUT: B, N, D
    batch_size = input.shape[0]
    return torch.cat([input[i, index[i,:,0], index[i][:,1], :].unsqueeze(0) \
        for i in range(batch_size)], dim = 0)

def array_info(arr, anno):
    print(colored(anno, 'yellow'), arr.shape, arr.max(), arr.min())

def normalize_to_img(arr): #tensor to img
    
    if torch.is_tensor(arr):
        a = arr.detach().cpu().numpy()

    if a.shape[0]<20:
        a = a[0]

    if a.shape[-1]>20:
        a = a[:,:,None]
    
    if a.shape[-1] != 3:
        a = a.repeat(1,1,3)
    
    a /= a.max()
    
    return a*255


def scatter_high_dimension(src, input, index, dim=1):
    
    # input: B, N, V
    # index: B, M, K,   M,K samples from N
    # output: B, M, K, V

    #index, B, M, K, X?
    #output: B, M, K, X, V 

    tgt_idx_shape = list(index.shape)
    tgt_idx_shape = tgt_idx_shape + [input.shape[2]]

    idx_shape_remain = index[0].numel()

    #index from B, M, K, to B, M*K
    index = index.view(index.shape[0], index[0].numel())

    #index to B, M*K, V
    index = torch.repeat_interleave(index, repeats = input.shape[2], dim=1).view(index.shape[0], idx_shape_remain, input.shape[2])

    #s0 = torch.zeros_like(input).to(input.device)
    return src.scatter_(dim=dim, index = index, src= input)
    

def index_high_dimension(input, index, dim=1, update = False):
    
    # input: B, N, V
    # index: B, M, K,   M,K samples from N
    # output: B, M, K, V

    #index, B, M, K, X?
    #output: B, M, K, X, V 

    tgt_idx_shape = list(index.shape)
    tgt_idx_shape = tgt_idx_shape + [input.shape[2]]

    idx_shape_remain = index[0].numel()

    #index from B, M, K, to B, M*K
    index = index.view(index.shape[0], index[0].numel())

    #index to B, M*K, V
    index = torch.repeat_interleave(index, repeats = input.shape[2], dim=1).view(index.shape[0], idx_shape_remain, input.shape[2])

    if update:        
        return input.gather(dim=dim, index = index).view(tgt_idx_shape)
    else:
        return torch.gather(input=input, dim=dim, index = index).view(tgt_idx_shape)

def vis_high_dimension_img(features_, compon=2, label = None, is_pca = False):
    if torch.is_tensor(features_):
        features = features_.cpu().numpy()
    else:
        features = features_
    
    feat_shape = features_.shape
    
    if is_pca: 
        tsne = PCA(n_components=compon).fit_transform(features.reshape(-1, feat_shape[-1]))
    else:
        tsne = TSNE(n_components=compon).fit_transform(features.reshape(-1, feat_shape[-1]))

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range
    
    for i in range(tsne.shape[-1]):
        tsne[:,i] = scale_to_01_range(tsne[:, i])
    
    new_shape = (*feat_shape[:-1], compon)     
    return tsne.reshape(new_shape)

import numpy as np
import cv2
import io


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    #img = Image.open(buf)
    img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), 1)
    return img
        
def plot_scat(map, uv_seg):
    colors = np.array(["beige", "red","green","blue","pink","orange","purple","yellow","brown","gray","cyan"])
    
    seg_body = {
        "0" : "bk",
        "1" : "head",
        "2" : "body",
        "3" : "hand",
        "4" : "hand",
        "5" : "foot",
        "6" : "foot",
        "7" : "arm",
        "8" : "arm",
        "9" : "leg",
        "10" : "leg"            
    }
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xy = np.floor(uv_seg[..., 0]).astype(np.int16)
    if map.ndim == 3:    
        for i in range(1,11):
            x,y = np.where(xy == i)
            ax.scatter(map[x,y, 0], map[x,y, 1], c = colors[i], label = seg_body["%d" % i])
    elif map.ndim == 2:
        for i in range(1,11):
            x = np.where(xy == i)
            ax.scatter(map[x, 0], map[x, 1], c = colors[i], label = seg_body["%d" % i])
                    
    ax.legend(loc="best")
    #plt.show()
    
    return fig2img(fig)



    