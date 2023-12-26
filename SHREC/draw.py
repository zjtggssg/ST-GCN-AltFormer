


import matplotlib.pyplot as plt
import numpy as np

class_names = ['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation CW', 'Rotation CCW', 'Swipe Right', 'Swipe Left', 'Swipe Up',
               'Swipe Down', 'Swipe X', 'Swipe +', 'Swipe V', 'Shake']

# Data
if __name__ == '__main__':
    data = np.array([[0.98, 0.93, 0.98, 0.90, 0.98, 0.98, 0.98, 0.98, 0.94, 0.98, 0.99, 1.00, 1.00, 1.00],
                     [0.97, 0.95, 0.98, 0.88, 1.00, 0.98, 1.00, 1.00, 0.96, 1.00, 0.99, 0.98, 1.00, 1.00]])

    # Find changes and mark "Joint&Bone" values in red
    changed_indices = np.where(data[0] != data[1])[0]
    red_indices = [class_names.index(name) for name in class_names if 'Joint&Bone' in name]

    # Plotting
    fig, ax = plt.subplots(dpi=300)
    bar_width = 0.35
    index = np.arange(len(class_names))

    bar1 = ax.bar(index, data[0], bar_width, label='Joint', color='gray')
    bar2 = ax.bar(index + bar_width, data[1], bar_width, label='Joint&Bone', color='blue')

    # Add labels and title
    ax.set_xlabel('Gestures')
    ax.set_ylabel('Scores')
    ax.set_title('Gesture Recognition Scores for Joint and Joint&Bone')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(class_names, rotation=90, ha='right')

    # Add numerical values on top of the bars for changed indices
    for i in changed_indices:
        height_joint = bar1[i].get_height()
        height_joint_bone = bar2[i].get_height()
        ax.text(bar1[i].get_x() + bar1[i].get_width() / 2, height_joint,
                f'{height_joint:.2f}', ha='center', va='bottom')
        ax.text(bar2[i].get_x() + bar2[i].get_width() / 2, height_joint_bone,
                f'{height_joint_bone:.2f}', ha='center', va='bottom')

    # Add numerical values on top of the bars for "Joint&Bone" in red
    for i in red_indices:
        height_joint_bone = bar2[i].get_height()
        ax.text(bar2[i].get_x() + bar2[i].get_width() / 2, height_joint_bone,
                f'{height_joint_bone:.2f}', ha='center', va='bottom', color='red')

    # Adjust legend position
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


    # Show the plot

    plt.show()
    plt.savefig('/data/zjt/gesture_recognition_scores.png')