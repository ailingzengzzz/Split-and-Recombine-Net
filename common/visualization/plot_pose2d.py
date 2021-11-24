import cv2
import numpy as np

# (R,G,B)
color1 = [(28,26,228),(28,26,228),(28,26,228),
          (74,175,77), (74,175,77),(74,175,77),
          (153,255,255),(153,255,255),(153,255,255),(153,255,255),
          (163,78,152),(163,78,152),(163,78,152),
          (0,127,255),(0,127,255),(0,127,255)]

link_pairs1 = [
    [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
     [5, 6], [0, 7],[7, 8], [8, 9], [9, 10],
     [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
]

point_color1 = [(0,0,0), (0,0,255), (0,0,255), (0,0,255),
                (0,255,0),(0,255,0),(0,255,0),
                (138,41,231),(138,41,231),(138,41,231),(138,41,231),
                (179,112,117),(179,112,117),(179,112,117),
                (2,95,217),(2,95,217),(2,95,217)]

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color
        self.line_color = []
        for i in range(len(self.color)):
            self.line_color.append(self.color[i])

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(self.point_color[i])

def show_2d_hm36_pose(img_path, pose_2d, index=0):
    # plot single pose from a image
    colorstyle = ColorStyle(color1, link_pairs1, point_color1)
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
    if img_path is None:
        img = np.zeros((1000, 1000,3), dtype=np.uint8)
    else:
        img = cv2.imread(img_path)

    kps = pose_2d  # 2d pose in pixel unit, shape [17, 2]
    for j, c in enumerate(connections):
        start = kps[c[0]]
        end = kps[c[1]]
        cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), colorstyle.line_color[j], 3)
        cv2.circle(img, (int(kps[j, 0]), int(kps[j, 1])), 4, colorstyle.ring_color[j], 2)
    cv2.imshow('3DPW Example', img)
    #cv2.imwrite('data/3dpw/validation/{}_{}_{:05d}.jpg'.format(seq, p_id, index), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
