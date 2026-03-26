from .VN_layers import VNLinearLeakyReLU, VNLeakyReLU, VNStdFeature
from .ops import index_select

def estimate_normals(knn_xyz):

    centroid = knn_xyz.mean(dim=1, keepdim=True)  # [N, 1, 3]
    centered = knn_xyz - centroid  
    cov_matrix = torch.bmm(centered.transpose(1, 2), centered)  
    _, _, v = torch.svd(cov_matrix)  
    normals = v[:, :, 2]  
    normals = normals.unsqueeze(1).expand(-1, knn_xyz.shape[1], -1)  # [N, K, 3]
    normals = F.normalize(normals, p=2, dim=-1)  
    return normals  # [N, K, 3]

class CorrelationNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], last_bn=False, temp=1):
        super(CorrelationNet, self).__init__()
        self.vn_layer = VNLinearLeakyReLU(in_channel, out_channel * 2, dim=4, share_nonlinearity=False, negative_slope=0.2)
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        self.temp = temp

        hidden_unit = list() if hidden_unit is None else copy.deepcopy(hidden_unit)
        hidden_unit.insert(0, out_channel * 2+2)
        hidden_unit.append(out_channel)
        for i in range(1, len(hidden_unit)):  # from 1st hidden to next hidden to last hidden
            self.mlp_convs_hidden.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1,
                                                   bias=False if i < len(hidden_unit) - 1 else not last_bn))
            if i < len(hidden_unit) - 1 or last_bn:
                self.mlp_bns_hidden.append(nn.BatchNorm1d(hidden_unit[i]))

    def forward(self, xyz, scalars=None):
        # xyz : N * D * 3 * k
        N, _, _, K = xyz.size()
        scores = self.vn_layer(xyz)  # N,8,3,k
        scores = torch.norm(scores, p=2, dim=2)  # transform rotation equivairant feats into rotation invariant feats
        # N,8,k
        if scalars is not None:
            scores = torch.cat([scores, scalars], dim=1) # N,10,k
        for i, conv in enumerate(self.mlp_convs_hidden):
            if i < len(self.mlp_convs_hidden) - 1:
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))  # N,4,k
            else:  # if the output layer, no ReLU
                scores = conv(scores)
                if self.last_bn:
                    scores = self.mlp_bns_hidden[i](scores)
        scores = F.softmax(scores/self.temp, dim=1)
        return scores


class MG_Conv_Block(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, share_nonlinearity=False):
        super(MG_Conv_Block, self).__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(in_channel=3, out_channel=self.kernel_size, hidden_unit=[self.kernel_size])

        in_dim = in_dim + 2   # 1 + 2: [xyz, mean, cross]
        tensor1 = nn.init.kaiming_normal_(torch.empty(self.kernel_size, in_dim, out_dim // 2)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(in_dim, self.kernel_size * out_dim // 2)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)  # 3,4*32/2=64

        self.relu = VNLeakyReLU(out_dim//2, share_nonlinearity)
        self.unary = VNLinearLeakyReLU(out_dim//2, out_dim)


    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):

        N, K = neighbor_indices.shape

        # compute relative coordinates
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)  
        centers = pts.mean(-1, keepdim=True).repeat(1, 1, 1, K)
        cross = torch.cross(pts, centers, dim=2)
        local_feats = torch.cat([pts, centers, cross], 1) 
        knn_xyz = s_pts[neighbor_indices]  
        normals = estimate_normals(knn_xyz)  
        center_dir = centers.squeeze(1).transpose(1, 2) 
        dot_product = torch.sum(normals * center_dir, dim=-1, keepdim=True)  
        centered = knn_xyz - knn_xyz.mean(dim=1, keepdim=True)
        cov_matrix = torch.bmm(centered.transpose(1, 2), centered)
        _, s, _ = torch.svd(cov_matrix)
        curvature = s.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(dot_product)  
        scalars = torch.cat([dot_product, curvature], dim=-1)  
        scalars = scalars.permute(0, 2, 1) 
        scores = self.score_net(local_feats, scalars) 
    
        # use correlation scores to assemble features
        pro_feats = torch.einsum('ncdk,cf->nfdk', local_feats, self.weightbank)  
        pro_feats = pro_feats.reshape(N,  self.kernel_size, -1, 3, K)  
        pro_feats = (pro_feats * scores[:, :, None, None]).sum(1)
        normed_feats = F.normalize(pro_feats, p=2, dim=2) 
        # mean pooling
        new_feats = normed_feats.mean(-1) # [N, D/2=16, 3]
        # applying VN ReLU after pooling to reduce computation cost
        new_feats = self.relu(new_feats)
        # mapping D/2 -> D
        new_feats = self.unary(new_feats)  # [N, D, 3]

        return new_feats

class PARE_Conv_Resblock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, shortcut_linear=False, share_nonlinearity=False, conv_info=None):
        super(PARE_Conv_Resblock, self).__init__()
        self.kernel_size = kernel_size
        self.score_net = CorrelationNet(in_channel=3, out_channel=self.kernel_size, hidden_unit=[self.kernel_size])

        self.conv_way = conv_info["conv_way"]
        self.use_xyz = conv_info["use_xyz"]
        conv_dim = in_dim * 2 if self.conv_way == 'edge_conv' else in_dim
        if self.use_xyz: conv_dim += 1
        tensor1 = nn.init.kaiming_normal_(torch.empty(self.kernel_size, conv_dim, out_dim//2)).contiguous()
        tensor1 = tensor1.permute(1, 0, 2).reshape(conv_dim, self.kernel_size * out_dim//2)
        self.weightbank = nn.Parameter(tensor1, requires_grad=True)

        self.relu = VNLeakyReLU(out_dim//2, share_nonlinearity)
        # self.shortcut_proj = VNLinear(in_dim, out_dim) if shortcut_linear else nn.Identity()
        self.shortcut_proj = nn.Sequential(
            VNLinear(in_dim, out_dim//2),
            VNLeakyReLU(out_dim//2, share_nonlinearity),
            VNLinear(out_dim//2, out_dim)
        ) if shortcut_linear else nn.Identity()
        self.unary = VNLinearLeakyReLU(out_dim//2, out_dim)

        # self.norm_embedding = VNLinear(1, out_dim//2)
        # self.curv_embedding = VNLinear(1, out_dim//2)
        # self.maa = MAA(in_channels=out_dim//2, features_num=3)

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):

        N, K = neighbor_indices.shape
        pts = (s_pts[neighbor_indices] - q_pts[:, None]).unsqueeze(1).permute(0, 1, 3, 2)  
        # compute relative coordinates
        center = pts.mean(-1, keepdim=True).repeat(1, 1, 1, K)
        cross = torch.cross(pts, center, dim=2)
        local_feats = torch.cat([pts, center, cross], 1)
        knn_xyz = s_pts[neighbor_indices]  
        normals = estimate_normals(knn_xyz)  
        center_dir = center.squeeze(1).transpose(1, 2)  
        dot_product = torch.sum(normals * center_dir, dim=-1, keepdim=True)
        centered = knn_xyz - knn_xyz.mean(dim=1, keepdim=True)
        cov_matrix = torch.bmm(centered.transpose(1, 2), centered)
        _, s, _ = torch.svd(cov_matrix)
        curvature = s.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(dot_product)  

        scalars = torch.cat([dot_product, curvature], dim=-1)  
        scalars = scalars.permute(0, 2, 1)  
        scores = self.score_net(local_feats, scalars)
        # dot_product = dot_product.permute(0, 2, 1)  
        # curvature = curvature.permute(0, 2, 1)  # [N, 1, K]
        # scores = self.score_net(local_feats, dot_product, curvature) # [N, kernel_size,  K]  

        # gather neighbors features
        neighbor_feats = s_feats[neighbor_indices, :].permute(0, 2, 3, 1)  
        # shortcut
        identify = neighbor_feats[..., 0]  # [N, in_dim, 3]
        identify = self.shortcut_proj(identify)  # [N, out_dim, 3]
        # get edge features
        if self.conv_way == 'edge_conv':
            q_feats = neighbor_feats[..., 0:1] # [N, in_dim, 3, 1]
            neighbor_feats = torch.cat([neighbor_feats - q_feats, neighbor_feats], 1)
        # use relative coordinates
        if self.use_xyz:
            neighbor_feats = torch.cat([neighbor_feats, pts], 1)# [N, 32+32+1, 3, 1]
        # use correlation scores to assemble features
        pro_feats = torch.einsum('ncdk,cf->nfdk', neighbor_feats, self.weightbank)
        pro_feats = pro_feats.reshape(N, self.kernel_size, -1, 3, K)
        pro_feats = (pro_feats * scores[:, :, None, None]).sum(1)

        normed_feats = F.normalize(pro_feats, p=2, dim=2)

        new_feats = normed_feats.mean(-1)
        new_feats = self.relu(new_feats)
        new_feats = self.unary(new_feats)  # [N, D, 3]
        new_feats = new_feats + identify
        return new_feats