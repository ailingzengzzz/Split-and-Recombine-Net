### Many thanks to: https://raw.githubusercontent.com/bastianwandt/RepNet/7b9185cadd12f850e9fa1754505fca68c34be4ed/plot17j.py
### Changed the index to Human3.6M 17 keypoints
import numpy as np

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import axes3d, Axes3D

def plot17j(poses,  ax=None, subject=None, action=None, show_animation=False):
    if not show_animation:
        plot_idx = 1
        if len(poses.shape)>2:
            fig = plt.figure()
            frames = np.linspace(start=0, stop=poses.shape[0]-1, num=6).astype(int)
            for i in frames:
                ax = fig.add_subplot(2, 3, plot_idx, projection='3d')
                pose = poses[i]
                x = pose[:, 0]
                y = pose[:, 1]
                z = pose[:, 2]
                ax.scatter(x, y, z)
                ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])])
                ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
                ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])])
                ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])])
                ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
                ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])])
                ax.plot(x[([0, 7])], y[([0, 7])], z[([0, 7])])
                ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
                ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
                ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])])
                ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])])
                ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
                ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])])
                ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])])
                ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])
                ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])])
                # Create cubic bounding box to simulate equal aspect ratio
                max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
                Xb = 0.5 * max_range * np.mgrid[-1:1:1, -1:1:1, -1:1:1][0].flatten() + 0.5 * (x.max() + x.min())
                Yb = 0.5 * max_range * np.mgrid[-1:1:1, -1:1:1, -1:1:1][1].flatten() + 0.5 * (y.max() + y.min())
                Zb = 0.5 * max_range * np.mgrid[-1:1:1, -1:1:1, -1:1:1][2].flatten() + 0.5 * (z.max() + z.min())

                for xb, yb, zb in zip(Xb, Yb, Zb):
                    ax.plot([xb], [yb], [zb], 'w')
                radius = 2
                ax.view_init(elev=75, azim=110)
                ax.set_xlim3d([-radius / 2, radius / 2])
                ax.set_zlim3d([0, radius])
                ax.set_ylim3d([-radius / 2, radius / 2])

                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                # ax.invert_zaxis()
                ax.axis('equal')
                # ax.axis('off')

                # ax.set_title('camera = ' + str(i))

                plot_idx += 1
        else:
            pose = poses
            x = pose[:, 0]
            y = pose[:, 1]
            z = pose[:, 2]
            ax.scatter(x, y, z)
            ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])])
            ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])])
            ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])])
            ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])])
            ax.plot(x[([0, 7])], y[([0, 7])], z[([0, 7])])
            ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])])
            ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])])
            ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
            ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])])
            ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])])
            ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])
            ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])])
            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:1:1, -1:1:1, -1:1:1][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:1:1, -1:1:1, -1:1:1][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:1:1, -1:1:1, -1:1:1][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')
            # radius = 2
            # ax.view_init(elev=15., azim=110)
            # ax.set_xlim3d([-radius / 2, radius / 2])
            # ax.set_zlim3d([0, radius])
            # ax.set_ylim3d([-radius / 2, radius / 2])

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # ax.invert_zaxis()
            ax.axis('equal')
            #ax.axis('off')

            #ax.set_title('camera = ' + str(i))

            plot_idx += 1

        # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.show()
        plt.savefig('show/mean_train_pose_{}_{}'.format(subject, action), bbox_inches='tight')
        plt.close()

    else:
        def update(i):

            ax.clear()

            pose = poses[i]

            x = pose[:, 0]
            y = pose[:, 1]
            z = pose[:, 2]
            ax.scatter(x, y, z)

            ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])])
            ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])])
            ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])])
            ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])])
            ax.plot(x[([0, 7])], y[([0, 7])], z[([0, 7])])
            ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])])
            ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])])
            ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
            ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])])
            ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])])
            ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])
            ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            plt.axis('equal')

        a = anim.FuncAnimation(fig, update, frames=len(poses), repeat=False)
        plt.show()
        plt.savefig('show/mean_train_pose_{}_{}'.format(subject, action), bbox_inches='tight')
        plt.close()

    return


def drawskeleton(img, kps, thickness=3, lcolor=(255,0,0), rcolor=(0,0,255), mpii=2):

    if mpii == 0: # h36m with mpii joints
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif mpii == 1: # only mpii
        connections = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                       [7, 8], [8, 9], [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [14, 15]]
        LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
    else: # default h36m
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)


def show3Dpose(channels, ax, radius=40, mpii=2, lcolor='#ff0000', rcolor='#0000ff'):
    vals = channels

    if mpii == 0: # h36m with mpii joints
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    elif mpii == 1: # only mpii
        connections = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                       [7, 8], [8, 9], [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [14, 15]]
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    else: # default h36m
        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    for ind, (i,j) in enumerate(connections):
        x, y, z = [np.array([vals[i, c], vals[j, c]]) for c in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[ind] else rcolor)

    RADIUS = radius  # space around the subject
    if mpii == 1:
        xroot, yroot, zroot = vals[6, 0], vals[6, 1], vals[6, 2]
    else:
        xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
