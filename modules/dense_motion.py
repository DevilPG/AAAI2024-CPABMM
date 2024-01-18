from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d
import math
from libcpab.cpab import Cpab

# pre-defined tessellation size and corresponding dimension of CPAB with/without perserving volume constraint
tes2d_perserveV = {
    3:11,
    4:15,
    5:19,
    6:27,
    7:35,
    8:47,
    9:59,
    10:75,
}
tes2d_noConstraint = {
    3:34, #24
    4:58, #32
    5:90, #40
    6:130, # 48
    7:178,
    8:234,
    9:298,
    10:370,
}


class CPABTransformation(nn.Module):
    """
    Keypoint-Based Transformation Inference in Section 3.2 in the paper
    volume perservation constraint: whether to preverve volume in the process of transformation
    """
    def __init__(self, tes_size=[4,4], zero_boundary=False, volume_perservation=False, max_iter=100) -> None:
        super(CPABTransformation, self).__init__()
        self.T = Cpab(tes_size, backend='pytorch', device='gpu', 
             zero_boundary=zero_boundary, volume_perservation=volume_perservation, override=False)
        if volume_perservation:
            self.d = tes2d_perserveV[tes_size[0]]
        else:
            self.d = tes2d_noConstraint[tes_size[0]]
        self.max_iter = max_iter

        
    def forward(self, source_image, kp_source, kp_driving, num_of_pts, max_iter = -1):
        if max_iter!=-1:
            Max_iter = max_iter
        else:
            Max_iter = self.max_iter
        bs, _, h, w = source_image.shape
        kp_s = kp_source.detach().view(bs, -1, num_of_pts, 2)
        kp_s = kp_s.view(-1, num_of_pts, 2).permute(0,2,1) # B*num_cpab,2,5
        kp_d = kp_driving.detach().view(bs, -1, num_of_pts, 2)
        kp_d = kp_d.view(-1, num_of_pts, 2).permute(0,2,1) # B*num_cpab,2,5

        kp_grid_d = (kp_d+1)/2
        BS = kp_s.shape[0]
        theta = self.T.identity(BS, epsilon=1e-6)
        theta.requires_grad = True
        # theta = torch.autograd.Variable(self.T.identity(BS, epsilon=1e-6), requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=1e-2)
        for i in range(int(Max_iter)):
            optimizer.zero_grad()
            trans_kp = self.T.transform_grid(kp_grid_d, theta)
            trans_kp = 2*trans_kp-1
            # L_kp, Eq.(6) in the paper
            loss = torch.norm(trans_kp-kp_s)
            loss.backward(retain_graph=True)
            optimizer.step()
        
        grid = make_coordinate_grid((h,w), source_image.type())
        grid = (grid+1)/2 #[-1,1] -> [0,1]
        grid = grid.view(-1,2).permute(1,0)
        grid_t = self.T.transform_grid(grid, theta)
        grid_t = 2*grid_t - 1
        deformations = grid_t.permute(0,2,1).view(-1,h,w,2).view(bs,-1,h,w,2)
        return deformations
    
# 10 partial CPAB transformation, each have 5 control points + 1 global CPAB transformation
class DenseMotionNetwork(nn.Module):
    """
    Module that estimating an optical flow and multi-resolution occlusion masks 
                        from K+1 CPAB transformations and an affine transformation.
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_cpab, num_channels, 
                 scale_factor=0.25, bg = False, multi_mask = True, max_iter = 100, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()

        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        self.CPAB_transformation_partial = CPABTransformation(tes_size=[4,4], zero_boundary=False, volume_perservation=False, max_iter=max_iter)
        self.CPAB_transformation_global = CPABTransformation(tes_size=[6,6], zero_boundary=False, volume_perservation=False, max_iter=1.5*max_iter)

        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_channels * (num_cpab+1+1) + num_cpab*5+1),
                                   max_features=max_features, num_blocks=num_blocks)

        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(hourglass_output_size[-1], num_cpab + 1 + 1, kernel_size=(7, 7), padding=(3, 3))

        if multi_mask:
            up = []
            self.up_nums = int(math.log(1/scale_factor, 2))
            self.occlusion_num = 4
            
            channel = [hourglass_output_size[-1]//(2**i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i]//2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            channel = [hourglass_output_size[-i-1] for i in range(self.occlusion_num-self.up_nums)[::-1]]
            for i in range(self.up_nums):
                channel.append(hourglass_output_size[-1]//(2**(i+1)))
            occlusion = []
            
            for i in range(self.occlusion_num):
                occlusion.append(nn.Conv2d(channel[i], 1, kernel_size=(7, 7), padding=(3, 3)))
            self.occlusion = nn.ModuleList(occlusion)
        else:
            occlusion = [nn.Conv2d(hourglass_output_size[-1], 1, kernel_size=(7, 7), padding=(3, 3))]
            self.occlusion = nn.ModuleList(occlusion)

        self.num_cpab = num_cpab
        self.bg = bg
        self.kp_variance = kp_variance

        
    def create_heatmap_representations(self, source_image, kp_driving, kp_source):

        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_source = kp2gaussian(kp_source['fg_kp'], spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_driving - gaussian_source

        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type()).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)

        return heatmap

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param, max_iter = -1):
        bs, _, h, w = source_image.shape
        source_kp = kp_source['fg_kp']
        driving_kp = kp_driving['fg_kp'] 
        driving_to_source = self.CPAB_transformation_partial(source_image,source_kp,driving_kp, 5, max_iter)
        driving_to_source_global = self.CPAB_transformation_global(source_image,source_kp,driving_kp, 5*self.num_cpab, max_iter)

        identity_grid = make_coordinate_grid((h, w), type=source_kp.type()).to(source_kp.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # affine background transformation
        if not (bg_param is None):            
            identity_grid = to_homogeneous(identity_grid)
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
            identity_grid = from_homogeneous(identity_grid)

        transformations = torch.cat([identity_grid, driving_to_source, driving_to_source_global], dim=1)
        return transformations

    def create_deformed_source_image(self, source_image, transformations):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_cpab + 1+1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_cpab + 1+1), -1, h, w)
        transformations = transformations.view((bs * (self.num_cpab + 1+1), h, w, -1))
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, self.num_cpab+1+1, -1, h, w))
        return deformed

    def dropout_softmax(self, X, P):
        '''
        Dropout for CPAB transformations.
        '''
        drop = (torch.rand(X.shape[0],X.shape[1]) < (1-P)).type(X.type()).to(X.device)
        drop[..., 0] = 1
        drop = drop.repeat(X.shape[2],X.shape[3],1,1).permute(2,3,0,1)

        maxx = X.max(1).values.unsqueeze_(1)
        X = X - maxx
        X_exp = X.exp()
        X[:,1:,...] /= (1-P)
        mask_bool =(drop == 0)
        X_exp = X_exp.masked_fill(mask_bool, 0)
        partition = X_exp.sum(dim=1, keepdim=True) + 1e-6
        return X_exp / partition  

    def forward(self, source_image, kp_driving, kp_source, bg_param = None, dropout_flag=False, dropout_p = 0, max_iter = -1):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param, max_iter)
        deformed_source = self.create_deformed_source_image(source_image, transformations)
        out_dict['deformed_source'] = deformed_source
        # out_dict['transformations'] = transformations
        deformed_source = deformed_source.view(bs,-1,h,w)
        input = torch.cat([heatmap_representation, deformed_source], dim=1)
        input = input.view(bs, -1, h, w)

        prediction = self.hourglass(input, mode = 1)

        contribution_maps = self.maps(prediction[-1]) 
        if(dropout_flag):
            contribution_maps = self.dropout_softmax(contribution_maps, dropout_p)
        else:
            contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict['contribution_maps'] = contribution_maps

        # Combine the N+2 transformations
        # Eq(7) in the paper
        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)
        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation # Optical Flow

        occlusion_map = []
        if self.multi_mask:
            for i in range(self.occlusion_num-self.up_nums):
                occlusion_map.append(torch.sigmoid(self.occlusion[i](prediction[self.up_nums-self.occlusion_num+i])))
            prediction = prediction[-1]
            for i in range(self.up_nums):
                prediction = self.up[i](prediction)
                occlusion_map.append(torch.sigmoid(self.occlusion[i+self.occlusion_num-self.up_nums](prediction)))
        else:
            occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))
                
        out_dict['occlusion_map'] = occlusion_map # Multi-resolution Occlusion Masks
        return out_dict