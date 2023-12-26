import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
from data_process.Hand_Dataset import Hand_Dataset
from model.ST_GCN_AltFormer import ST_GCN_AltFormer
from model.ST_GCN_Altformer_new import ST_GCN_AltFormer_new
from model.ST_GCN_Trans import ST_GCN_Trans
from model.ST_TR.ST_TR_new import ST_TR_new

output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_model_st_ts(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_AltFormer(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
                             only_attention=True,
                             tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
                             relative=False,
                             double_channel=True,
                             drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10, dim_block2=30,
                             dim_block3=75,
                             data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                             kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4, device=output_device,
                             graph=address,
                             graph_args=graph_args
                             )

    model = torch.nn.DataParallel(model).to(output_device)
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.975.pth'
    state_dict = torch.load(model_path, map_location=output_device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def init_model_str(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_TR_new(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
                      only_attention=True,
                      tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
                      relative=False,
                      double_channel=True,
                      drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10, dim_block2=30,
                      dim_block3=75,
                      data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                      kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4, device=output_device,
                      graph=address,
                      graph_args=graph_args
                      )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/str.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model

def init_model_vit(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_Trans(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
                      only_attention=True,
                      tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
                      relative=False,
                      double_channel=True,
                      drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10, dim_block2=30,
                      dim_block3=75,
                      data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                      kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4, device=output_device,
                      graph=address,
                      graph_args=graph_args
                      )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/vit.pth'
    state_dict = torch.load(model_path)
    checkpoint = {k: v.shape for k, v in state_dict.items() if k in state_dict}
    print(checkpoint)
    model.load_state_dict(state_dict)

    model.eval()
    return model

def init_model_st(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_AltFormer(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
                             only_attention=True,
                             tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
                             relative=False,
                             double_channel=True,
                             drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10, dim_block2=30,
                             dim_block3=75,
                             data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                             kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4, device=output_device,
                             graph=address,
                             graph_args=graph_args
                             )

    model = torch.nn.DataParallel(model).to(output_device)
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.9631.pth'
    state_dict = torch.load(model_path, map_location=output_device)

    model.load_state_dict(state_dict)

    model.eval()
    return model


def init_model_ttr(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_TR_new(channel=3, num_class=class_num, window_size=300, num_point=22, attention=False,
                      only_attention=True,
                      tcn_attention=True, all_layers=False, only_temporal_attention=True, attention_3=False,
                      relative=False,
                      double_channel=True,
                      drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10, dim_block2=30,
                      dim_block3=75,
                      data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                      kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4, device=output_device,
                      graph=address,
                      graph_args=graph_args
                      )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/ttr.pth'

    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model


def init_model_ST_TS_new(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_AltFormer_new(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
                      only_attention=True,
                      tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
                      relative=False,
                      double_channel=True,
                      drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10, dim_block2=30,
                      dim_block3=75,
                      data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                      kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4, device=output_device,
                      graph=address,
                      graph_args=graph_args
                      )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/test2.pth'
    state_dict = torch.load(model_path)
    checkpoint = {k: v.shape for k, v in state_dict.items() if k in state_dict}
    print(checkpoint)
    model.load_state_dict(state_dict)

    model.eval()
    return model

if __name__ == '__main__':
    print('==> Building model..')

    all_data = np.load('/data/zjt/HandGestureDataset_SHREC2017/gesture/test14.npy', allow_pickle=True)

    model = init_model_ST_TS_new(0)

    # Move the model to cuda:0
    model = model.to(output_device)

    all_dataset = Hand_Dataset(all_data, use_data_aug=False, time_len=180, expand=0)

    # Move dummy_input to cuda:0
    dummy_input = all_dataset[0]['skeleton'].unsqueeze(0).to(output_device)
    print(dummy_input.shape)
    print(type(dummy_input))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", params)

    print(dummy_input.device)
    print(torch.cuda.current_device())

    # Ensure the model and input are on the same device
    model.to(output_device)
    dummy_input = dummy_input.to(output_device)
    score = model(dummy_input)
    print(score)
    flops = FlopCountAnalysis(model, (dummy_input,))
    print("FLOPs: ", flops.total())

    print(parameter_count_table(model))
    # flops, params = profile(model, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
