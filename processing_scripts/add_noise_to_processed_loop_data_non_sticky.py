import numpy as np

sim_timestep = 0.02
monomer_size_um = 21
step = 1

noise_um = 0
for loop_num in [3, 4, 5, 6, 7, 8]:
    for rep in range(10,20):
        mat = np.loadtxt(f'../simulation_data_processed/loop_{loop_num}_non_sticky_{rep}_step_{step}.txt')

        # rescale time and distance to seconds and um
        mat[:,0] = mat[:,0] * sim_timestep
        mat[:,1:7] = mat[:,1:7] * monomer_size_um

        # calculate the 3D distance
        dist_3D = np.sqrt(np.sum(np.square(mat[:,1:4]-mat[:,4:7]), 1))
        mat_with_dist_3D = np.concatenate((mat, dist_3D.reshape((-1,1))), axis=1)

        np.savetxt(f'../simulation_data_processed/loop_{loop_num}_non_sticky_{rep}_step_{step}_noise_{noise_um}um.txt', mat_with_dist_3D, fmt='%.2f')

for noise_um in [10, 20, 30, 40, 50]:
    for loop_num in [3, 4, 5, 6, 7, 8]:
        for rep in range(10,20):
            mat = np.loadtxt(f'../simulation_data_processed/loop_{loop_num}_non_sticky_{rep}_step_{step}.txt')

            # rescale time and distance to seconds and um
            mat[:,0] = mat[:,0] * sim_timestep
            mat[:,1:7] = mat[:,1:7] * monomer_size_um

            # add noise (2*sigma for z-direction)
            mat[:,1] = mat[:,1] + np.random.normal(0, noise_um, mat.shape[0])  # x1
            mat[:,2] = mat[:,2] + np.random.normal(0, noise_um, mat.shape[0])  # y1
            mat[:,3] = mat[:,3] + np.random.normal(0, noise_um*2, mat.shape[0])  # z1
            mat[:,4] = mat[:,4] + np.random.normal(0, noise_um, mat.shape[0])  # x2
            mat[:,5] = mat[:,5] + np.random.normal(0, noise_um, mat.shape[0])  # y2
            mat[:,6] = mat[:,6] + np.random.normal(0, noise_um*2, mat.shape[0])  # z2

            # calculate the 3D distance
            dist_3D = np.sqrt(np.sum(np.square(mat[:,1:4]-mat[:,4:7]), 1))
            mat_with_dist_3D = np.concatenate((mat, dist_3D.reshape((-1,1))), axis=1)

            np.savetxt(f'../simulation_data_processed/loop_{loop_num}_non_sticky_{rep}_step_{step}_noise_{noise_um}um.txt', mat_with_dist_3D, fmt='%.2f')
