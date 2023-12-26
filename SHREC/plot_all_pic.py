import numpy as np
from matplotlib import pyplot as plt, rcParams


def percentile_matrix(normalized_matrix):

    percentile_80 = np.percentile(normalized_matrix, 80)
    normalized_matrix[normalized_matrix <= percentile_80] = 0

    return normalized_matrix


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

# Customize further as needed
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
def normalize_matrix(matrix,max,min):
    normalized =  matrix
    normalized_matrix = (normalized - min) / (max - min)

    return normalized_matrix

def get_min_max(st_matrix,ts_matrix,st_ts_matrix):
    st_max_value = np.max(st_matrix)
    print(st_max_value)
    ts_max_value = np.max(ts_matrix)
    print(ts_max_value)
    stts_max_value = np.max(st_ts_matrix)
    print(stts_max_value)
    global_max_value = np.max([st_max_value, ts_max_value, stts_max_value])
    # global_max_value = round(global_max_value, 2)
    print("Global max value across all matrices:", global_max_value)

    st_min_value = np.min(st_matrix)
    ts_min_value = np.min(ts_matrix)
    stts_min_value = np.min(st_ts_matrix)
    global_min_value = np.min([st_min_value, ts_min_value, stts_min_value])
    # global_min_value = round(global_min_value, 2)
    print("Global min value across all matrices:", global_min_value)

    return global_max_value,global_min_value


def plot_cam_pic(st, ts, st_ts, style, pd=None, selected_index=None):
    st_matrix = np.abs(st)

    ts_matrix = np.abs(ts)

    st_ts_matrix = np.abs(st_ts)

    global_max_value, global_min_value = get_min_max(st_matrix,ts_matrix,st_ts_matrix)

    st_matrix = normalize_matrix(st_matrix,global_max_value, global_min_value )
    ts_matrix = normalize_matrix(ts_matrix,global_max_value, global_min_value )

    st_ts_matrix = normalize_matrix(st_ts_matrix,global_max_value, global_min_value )

    global_max_value, global_min_value = get_min_max(st_matrix, ts_matrix, st_ts_matrix)
    # st_matrix = percentile_matrix(st_matrix)
    # ts_matrix = percentile_matrix(ts_matrix)
    # st_ts_matrix = percentile_matrix(st_ts_matrix)

    # Create a subplot with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    # Plot st_matrix
    im0 = axs[0].imshow(st_matrix.T, vmin=global_min_value,vmax=global_max_value, cmap='viridis', interpolation='none', aspect='auto')
    axs[0].set_title('ST-branch',fontname="DejaVu Sans")
    axs[0].set_xlabel("Frames",fontname="DejaVu Sans")
    axs[0].set_ylabel("Joints",fontname="DejaVu Sans")
    axs[0].set_yticks(np.arange(0, st_matrix.shape[1], 1))


    # Plot ts_matrix
    im1 = axs[1].imshow(ts_matrix.T,vmin=global_min_value,vmax=global_max_value, cmap='viridis', interpolation='none', aspect='auto')
    axs[1].set_title('TS-branch')
    axs[1].set_xlabel("Frames")
    axs[1].set_ylabel("Joints")
    axs[1].set_yticks(np.arange(0, ts_matrix.shape[1], 1))


    # Plot st_ts_matrix
    im2 = axs[2].imshow(st_ts_matrix.T, vmin=global_min_value,vmax=global_max_value,cmap='viridis', interpolation='none', aspect='auto')
    axs[2].set_title('AltFormer')
    axs[2].set_xlabel("Frames")
    axs[2].set_ylabel("Joints")
    axs[2].set_yticks(np.arange(0, st_ts_matrix.shape[1], 1))



    # Create a colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    plt.colorbar(im2, cax=cbar_ax)


    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3)


    plt.savefig('/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.png'.format(pd, style, selected_index, pd))




def plot_cam_muti_pic(joint, bone, style, pd=None, selected_index=None):
    st_matrix = np.abs(st)

    ts_matrix = np.abs(ts)

    st_ts_matrix = np.abs(st_ts)


    st_max_value = np.max(st_matrix)
    ts_max_value = np.max(ts_matrix)
    stts_max_value = np.max(st_ts_matrix)

    global_max_value = np.max([st_max_value, ts_max_value, stts_max_value])
    global_max_value = round(global_max_value, 2)
    print("Global max value across all matrices:", global_max_value)

    # st_matrix = percentile_matrix(st_matrix)
    # ts_matrix = percentile_matrix(ts_matrix)
    # st_ts_matrix = percentile_matrix(st_ts_matrix)

    # Create a subplot with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(30, 5))

    # Plot st_matrix
    im0 = axs[0].imshow(st_matrix.T, vmin = 0,vmax=global_max_value, cmap='viridis', interpolation='none', aspect='auto')
    axs[0].set_title('st_matrix',fontname="DejaVu Sans")
    axs[0].set_xlabel("frames",fontname="DejaVu Sans")
    axs[0].set_ylabel("joints",fontname="DejaVu Sans")
    axs[0].set_yticks(np.arange(0, st_matrix.shape[1], 1))


    # Plot ts_matrix
    im1 = axs[1].imshow(ts_matrix.T, cmap='viridis', vmin = 0,vmax=global_max_value,interpolation='none', aspect='auto')
    axs[1].set_title('ts_matrix')
    axs[1].set_xlabel("frames")
    axs[1].set_ylabel("joints")
    axs[1].set_yticks(np.arange(0, ts_matrix.shape[1], 1))


    # Plot st_ts_matrix
    im2 = axs[2].imshow(st_ts_matrix.T, cmap='viridis',vmin = 0,vmax=global_max_value, interpolation='none', aspect='auto')
    axs[2].set_title('st_ts_matrix')
    axs[2].set_xlabel("frames")
    axs[2].set_ylabel("joints")
    axs[2].set_yticks(np.arange(0, st_ts_matrix.shape[1], 1))



    # Create a colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar0 = plt.colorbar(im2, cax=cbar_ax, aspect='auto', ticks=np.linspace(0, global_max_value, 6))

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.4)


    plt.savefig('/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.png'.format(pd, style, selected_index, pd))



if __name__ == '__main__':

    # style_list = ['Swipe Left', 'Swipe up' ,'Expand','Swipe X','Swipe Down']
    style_list = ['Swipe X']
    # style_list = ['Swipe up', 'Swipe Left', 'Tap']

    # selected_index_list = [48,26,43,26,28]
    # selected_index_list = [26, 48, 44]
    selected_index_list = [26]
    for i in range(0,len(style_list)):
        style = style_list[i]
        selected_index = selected_index_list[i]
        emsemble = 'joint'
        pd = 'st_ts'

        pd1 = 'st'
        pd2 = 'ts'
        pd3 = 'bone'
        pd4 = 'motion'
        st = np.load(
            '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd1, style, selected_index,
                                                                 pd1))
        # print(st)
        ts = np.load(
            '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd2, style, selected_index,
                                                                                        pd2))

        # motion_matrix = np.load(
        #     '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd4, style, selected_index,
        #
        #                                                                             pd4))
        st_matrix = np.abs(st)

        ts_matrix = np.abs(ts)
        # emsemble_matrix = st_matrix +  ts_matrix
        emsemble_matrix = 0.5 * st_matrix + 0.5 * ts_matrix
        # emsemble_matrix = st_matrix + ts_matrix + bone_matrix + motion_matrix
        # emsemble_matrix = st_matrix + bone_matrix
        # st_ts_matrix =  st_matrix +  ts_matrix

        if emsemble == 'joint':
            # normalized_matrix = normalize(emsemble_matrix)
            np.save(
                '/data/zjt/HandGestureDataset_SHREC2017/gesture/mutil/{}/{}_{}_{}.npy'.format(emsemble, style,
                                                                                              selected_index, pd),
                emsemble_matrix)
            st_ts = emsemble_matrix
            plot_cam_pic(st, ts, st_ts, style, pd=pd, selected_index=selected_index)
        # else:
        #     bone_matrix = np.load(
        #         '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd3, style, selected_index,
        #                                                                                 pd3))
        #     # normalized_matrix = normalize(emsemble_matrix)
        #     # bone_matrix = normalize(bone_matrix)
        #
        #     mutil_matrix = np.concatenate(( emsemble_matrix, bone_matrix), axis=1)
        #     np.save(
        #         '/data/zjt/HandGestureDataset_SHREC2017/gesture/mutil/{}/{}_{}_{}.npy'.format(emsemble, style,
        #                                                                                       selected_index, pd),
        #         emsemble_matrix)
        #     plot_cam_pic_mutil(mutil_matrix, style, pd=pd, emsemble=emsemble)
