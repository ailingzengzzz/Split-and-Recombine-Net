# Split-and-Recombine-Net
Code for "SRNet: Improving Generalization in 3D Human Pose Estimation with a Split-and-Recombine Approach" ECCV'20

Human poses that are rare or unseen in a training set are challenging for a network to predict. Similar to the long-tailed distribution problem in visual recognition, the small number of examples for such poses limits the ability of networks to model them. Interestingly, local pose distributions suffer less from the long-tail problem, i.e., local
joint configurations within a rare pose may appear within other poses in the training set, making them less rare.
![observation](img/observation.png)

We propose to take advantage of this fact for better generalization to rare and unseen poses. To be specific, our method splits the body into local regions and processes them in
separate network branches, utilizing the property that a joint's position depends mainly on the joints within its local body region. Global coherence is maintained by recombining the global context from the rest of the body into each branch as a low-dimensional vector. With the reduced dimensionality of less relevant body areas, the training set distribution within network branches more closely reflects the statistics of local poses instead of global body poses, without sacrificing information important for joint inference. The proposed split-and-recombine approach, called SRNet, can be easily adapted to **both single-image and temporal models**, and it leads to appreciable improvements in the prediction of rare and unseen poses.
![framework](img/framework.png)

The comparison of Different network structures used for 2D to 3D pose estimation.
![comparison](img/comparison.png)


Coming soon!
