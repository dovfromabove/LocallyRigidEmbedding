import os
import numpy as np
from sklearn.metrics import euclidean_distances

from matplotlib import pyplot as plt
from matplotlib import colors

def load_matrix(filename, verbose=False):
    with open(filename, 'rb') as f:
        if verbose:
            print(f'File {filename} wwas loaded successfully')
        return np.load(f)

def plot_numerical_spectrum(eigenvalues, plot_coord, axs, cutoff=10, baseline=0, labels=True):
    def addlabels(x,y, ax, rotation=50, horizontalalignment='left', verticalalignment='bottom', rotation_mode='anchor'):
        for i in range(len(x)):
            ax.text(i,y[i],round(y[i], 5), rotation=rotation, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation_mode=rotation_mode)
    normalized_values = np.sort(np.abs(baseline-eigenvalues))[:cutoff]
    axs[plot_coord].bar(*zip(*list(enumerate(normalized_values))))
    if labels:
        addlabels(list(range(len(normalized_values))), normalized_values, axs[plot_coord])
    axs[plot_coord].set_xticks(np.arange(len(normalized_values)))


def plot_numerical_spectrums(evl_no_noise, evl_1p_noise, evl_10p_noise, filename, labels=False, k=None):
    fig, axs = plt.subplots(1, 3, figsize=(25,3))
    if k:
        fig.text(0.07, 0.48, rf'$k={k}$', fontsize=20, ha='center')
    plot_numerical_spectrum(evl_no_noise, 0, axs, labels=labels)
    plot_numerical_spectrum(evl_1p_noise, 1, axs, labels=labels)
    plot_numerical_spectrum(evl_10p_noise, 2, axs, labels=labels)
    plt.savefig(filename)
    print(f'Saved image of eigenvalue spectrum: {filename}')

def homogenous_transformation(array, degrees, translations):
    theta_x, theta_y, theta_z = np.radians(degrees)
    dx, dy, dz = translations

    rot_z = np.array([[np.cos(theta_z), -np.sin(theta_z),    0],
                        [np.sin(theta_z),  np.cos(theta_z),    0],
                        [0,                0,                  1]])

    rot_y = np.array([[np.cos(theta_y), 0, -np.sin(theta_y)],
                        [0,               1,                0],
                        [np.sin(theta_y), 0, np.cos(theta_y)]])

    rot_x = np.array([[1,  0,               0],
                        [0,  np.cos(theta_x), np.sin(theta_x)],
                        [0, -np.sin(theta_x), np.cos(theta_x)]])
    
    rot_matrix = rot_x @ rot_y @ rot_z
    translation_matrix = np.array([dx, dy, dz])

    transformation_matrix = np.hstack([rot_matrix, translation_matrix.reshape(3,1)])
    transformation_matrix = np.vstack([transformation_matrix, np.zeros(transformation_matrix.shape[1])])
    transformation_matrix[-1,-1] = 1
    modified_array = np.hstack([array, np.zeros((array.shape[0], 1)) ,np.ones((array.shape[0], 1))])
    return (modified_array @ transformation_matrix.T)[:,0:-2]

def embedding_error(ground_truth, embedding):
    return np.square(np.linalg.norm(euclidean_distances(ground_truth)  - euclidean_distances(embedding)))

def plot_embedding(R, subplot_coords, axs, title, x_coords=r"$\phi_1$", y_coords=r"$\phi_2$", plot_coordinates=[0,1], ref_coords=None, margin=0.05, grid=False, set_title=True):
    if ref_coords is None:
        ref_coords = R
    i = subplot_coords
    # i,j = subplot_coords
    norm = colors.Normalize(vmin=min(ref_coords[:, 0]), vmax=max(ref_coords[:, 0]))
    axs[i].scatter(R[:, plot_coordinates[0]], R[:, plot_coordinates[1]], c=ref_coords[:, 0], s=5, cmap=plt.cm.rainbow, norm=norm)
    if set_title:
        axs[i].set_title(title)
    axs[i].set(xlabel=x_coords, ylabel=y_coords)
    xmin, xmax, ymin, ymax = min(ref_coords[:, 0]), max(ref_coords[:, 0]), min(ref_coords[:, 1]), max(ref_coords[:, 1])
    axs[i].set_xlim([xmin-margin, xmax+margin])
    axs[i].set_ylim([ymin-margin, ymax+margin])
    if grid:
        axs[i].grid(visible='both', alpha=0.2)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)


def plot_embeddings(us_cities, r_no_noise, r_1p_noise, r_10p_noise, filename, margin=0.05, grid=False, k=None, plot_cities=True, set_title=True):
    if plot_cities:
        fig, axs = plt.subplots(1, 4, figsize=(25,4))
        i=1
        plot_embedding(us_cities, 0, axs, title='US Cities', x_coords=r'$x$', y_coords=r'$y$', margin=margin, grid=grid, set_title=set_title)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(22,4))
        i=0
    if k:
        fig.text(0.06, 0.48, rf'$k={k}$', fontsize=20, ha='center')
    plot_embedding(r_no_noise, i, axs, title='Locally Rigid Embedding: no noise', x_coords=r'$\phi_1$', y_coords=r'$\phi_2$', ref_coords=us_cities, margin=margin, grid=grid, set_title=set_title)
    plot_embedding(r_1p_noise, i+1, axs, title='Locally Rigid Embedding: 1% noise', x_coords=r'$\phi_1$', y_coords=r'$\phi_2$', ref_coords=us_cities, margin=margin, grid=grid, set_title=set_title)
    plot_embedding(r_10p_noise, i+2, axs, title='Locally Rigid Embedding: 10% noise', x_coords=r'$\phi_1$', y_coords=r'$\phi_2$', ref_coords=us_cities, margin=margin, grid=grid, set_title=set_title)
    plt.savefig(filename)
    print(f'Saved image of LRE results: {filename}')

def print_numerical_results(directory):
    us_cities = load_matrix(os.path.join(directory, 'us_cities.npy'))

    r_no_noise = load_matrix(os.path.join(directory,'r_no_noise.npy'))
    r_1p_noise = load_matrix(os.path.join(directory,'r_1%_noise.npy'))
    r_10p_noise = load_matrix(os.path.join(directory,'r_10%_noise.npy'))
    print(f"No noise embedding error ", embedding_error(us_cities, r_no_noise))
    print(f"1% noise embedding error ", embedding_error(us_cities, r_1p_noise))
    print(f"10% noise embedding error ", embedding_error(us_cities, r_10p_noise))

    eigenvalues_no_noise = load_matrix(os.path.join(directory,'evl_no_noise.npy'))
    eigenvalues_1p_noise = load_matrix(os.path.join(directory,'evl_1%_noise.npy'))
    eigenvalues_10p_noise = load_matrix(os.path.join(directory,'evl_10%_noise.npy'))
    print('No noise eigenvalues: ', np.sort(np.abs(eigenvalues_no_noise))[0:3])
    print('1% noise eigenvalues: ', np.sort(np.abs(eigenvalues_1p_noise))[0:4])
    print('10% noise eigenvalues: ', np.sort(np.abs(eigenvalues_10p_noise))[0:5])

    singular_values_no_noise = load_matrix(os.path.join(directory,'singular_val_no_noise.npy'))
    singular_values_1p_noise = load_matrix(os.path.join(directory,'singular_val_1%_noise.npy'))
    singular_values_10p_noise = load_matrix(os.path.join(directory,'singular_val_10%_noise.npy'))
    print('No noise singular values: ', singular_values_no_noise)
    print('1% noise singular values: ', singular_values_1p_noise)
    print('10% noise singular values: ', singular_values_10p_noise)


def show_results(k, directory):
    us_cities = load_matrix(os.path.join(directory, 'us_cities.npy'))

    spectrum_no_noise = load_matrix(os.path.join(directory,'evl_no_noise.npy'))
    spectrum_1p_noise = load_matrix(os.path.join(directory,'evl_1%_noise.npy'))
    spectrum_10p_noise = load_matrix(os.path.join(directory,'evl_10%_noise.npy'))
    plot_numerical_spectrums(spectrum_no_noise, spectrum_1p_noise, spectrum_10p_noise, os.path.join(directory,f'eval_spectrum_k{k}.png'), labels=False, k=k)

    r_no_noise = load_matrix(os.path.join(directory,'r_no_noise.npy'))
    r_1p_noise = load_matrix(os.path.join(directory,'r_1%_noise.npy'))
    r_10p_noise = load_matrix(os.path.join(directory,'r_10%_noise.npy'))
    r_no_noise = homogenous_transformation(r_no_noise, [0,0,-17.5], [0.04,0.020,0])
    r_1p_noise = homogenous_transformation(r_1p_noise, [180,0,7], [0.04,0.03,0])
    r_10p_noise = homogenous_transformation(r_10p_noise, [0,0,-2], [0.04,0.05,0])
    plot_embeddings(us_cities, r_no_noise, r_1p_noise, r_10p_noise, os.path.join(directory, f'LRE_k{k}.png'), grid=True, k=k, plot_cities=True, set_title=True)

if __name__ == '__main__':
    k = 8
    dir = f'output_070424_k{k}'
    show_results(k, dir)
    print_numerical_results(dir)
