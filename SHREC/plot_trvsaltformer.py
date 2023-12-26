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

def get_min_max(st_tr_matrix,st_ts_matrix):


    st_tr_max_value = np.max(st_tr_matrix)
    print(st_tr_max_value)
    stts_max_value = np.max(st_ts_matrix)
    print(stts_max_value)
    global_max_value = np.max([st_tr_max_value, stts_max_value])
    # global_max_value = round(global_max_value, 2)
    print("Global max value across all matrices:", global_max_value)

    st_tr_min_value = np.min(st_tr_matrix)
    stts_min_value = np.min(st_ts_matrix)
    global_min_value = np.min([st_tr_min_value, stts_min_value])
    # global_min_value = round(global_min_value, 2)
    print("Global min value across all matrices:", global_min_value)

    return global_max_value,global_min_value


def plot_cam_pic(st_tr, st_ts, style, pd=None, selected_index=None):
    st_tr_matrix  = np.abs(st_tr)
    st_ts_matrix = np.abs(st_ts)

    global_max_value, global_min_value = get_min_max(st_tr_matrix,st_ts_matrix)

    st_tr_matrix = normalize_matrix(st_tr_matrix,global_max_value, global_min_value )

    st_ts_matrix = normalize_matrix(st_ts_matrix,global_max_value, global_min_value )

    global_max_value, global_min_value = get_min_max(st_tr_matrix, st_ts_matrix)
    # st_matrix = percentile_matrix(st_matrix)
    # ts_matrix = percentile_matrix(ts_matrix)
    # st_ts_matrix = percentile_matrix(st_ts_matrix)

    # Create a subplot with 1 row and 3 columns
    fig, axs = plt.subplots(1, 2, figsize=(30, 5))

    # Plot st_matrix
    im0 = axs[0].imshow(st_tr_matrix.T, vmin=global_min_value,vmax=global_max_value, cmap='viridis', interpolation='none', aspect='auto')
    axs[0].set_title('ST-TR',fontname="DejaVu Sans")
    axs[0].set_xlabel("Frames",fontname="DejaVu Sans")
    axs[0].set_ylabel("Joints",fontname="DejaVu Sans")
    axs[0].set_yticks(np.arange(0, st_matrix.shape[1], 1))


    # Plot ts_matrix
    im1 = axs[1].imshow(st_ts_matrix.T,vmin=global_min_value,vmax=global_max_value, cmap='viridis', interpolation='none', aspect='auto')
    axs[1].set_title('AltFormer')
    axs[1].set_xlabel("Frames")
    axs[1].set_ylabel("Joints")
    axs[1].set_yticks(np.arange(0, ts_matrix.shape[1], 1))



    # Create a colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    plt.colorbar(im1, cax=cbar_ax)


    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3)


    plt.savefig('/data/zjt/HandGestureDataset_SHREC2017/gesture/vs/{}/{}_{}_{}.png'.format(pd, style, selected_index, pd))






if __name__ == '__main__':

    # style_list = ['Swipe Left', 'Swipe up' ,'Expand','Swipe X','Swipe Down']
    # style_list = ['Swipe X']
    style_list = ['Swipe Left']

    # selected_index_list = [48,26,43,26,28]
    selected_index_list = [49]
    # selected_index_list = [26]
    for i in range(0,len(style_list)):
        style = style_list[i]
        selected_index = selected_index_list[i]
        emsemble = 'joint'
        pd = 'st_ts'

        pd1 = 'st'
        pd2 = 'ts'
        pd3 = 'st_tr'

        st = np.load(
            '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd1, style, selected_index,
                                                                 pd1))
        # print(st)
        ts = np.load(
            '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd2, style, selected_index,
                                                                                        pd2))

        st_tr = np.load(
            '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}_new.npy'.format(pd3, style, selected_index,
                                                                                        pd3))

        # motion_matrix = np.load(
        #     '/data/zjt/HandGestureDataset_SHREC2017/gesture/{}/{}_{}_{}.npy'.format(pd4, style, selected_index,
        #
        #                                                                             pd4))
        st_matrix = np.abs(st)

        ts_matrix = np.abs(ts)

        st_tr_matrix = np.abs(st_tr)

        st_ts_matrix = 0.5 * st_matrix + 0.5 * ts_matrix


        plot_cam_pic(st_tr_matrix, st_ts_matrix, style, pd=pd, selected_index=selected_index)
