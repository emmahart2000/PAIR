import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random

def OODmetrics(b_in, b_out, x_in, x_out, b_encoder, b_decoder, x_encoder, x_decoder, inverse, forward):
    # Db(Eb(b))
    b_in_ae = (b_decoder.predict(b_encoder.predict(b_in))).squeeze()
    b_out_ae = (b_decoder.predict(b_encoder.predict(b_out))).squeeze()
    print(b_in_ae.shape)

    # Dx(Ex(x))
    x_in_ae = (x_decoder.predict(x_encoder.predict(x_in))).squeeze()
    x_out_ae = (x_decoder.predict(x_encoder.predict(x_out))).squeeze()
    print(x_in_ae.shape)

    # xhat = Dx(Mdag*Eb(b))
    x_in_pred = (x_decoder.predict((np.transpose(inverse @ np.transpose((b_encoder.predict(b_in)).reshape((b_in.shape[0], 147))))).reshape(b_in.shape[0], 7, 7, 3))).squeeze()
    x_out_pred = (x_decoder.predict((np.transpose(inverse @ np.transpose((b_encoder.predict(b_out)).reshape((b_out.shape[0], 147))))).reshape(b_out.shape[0], 7, 7, 3))).squeeze()
    print(x_in_pred.shape)

    # bhat = Db(M*E_x(xhat))
    b_in_pred = b_decoder.predict((np.transpose(forward @ np.transpose((b_encoder.predict(x_in_pred)).reshape((x_in_pred.shape[0], 147))))).reshape(x_in_pred.shape[0], 7, 7, 3))
    b_out_pred = b_decoder.predict((np.transpose(forward @ np.transpose((b_encoder.predict(x_out_pred)).reshape((x_out_pred.shape[0], 147))))).reshape(x_out_pred.shape[0], 7, 7, 3))
    print(b_in_pred.shape)

    # Dx(Ex(xhat))
    x_in_pred_ae = (x_decoder.predict(x_encoder.predict(x_in_pred))).squeeze()
    x_out_pred_ae = (x_decoder.predict(x_encoder.predict(x_out_pred))).squeeze()
    print(x_in_pred_ae.shape)

    # Ex(xhat)
    latent_x_in_pred = (x_encoder.predict(x_in_pred)).reshape(x_in_pred.shape[0], 147)
    latent_x_out_pred = (x_encoder.predict(x_out_pred)).reshape(x_out_pred.shape[0], 147)
    print(latent_x_in_pred.shape)

    # Mdag*Eb(b)
    latent_pred_x_in_pred = np.transpose(inverse @ np.transpose((b_encoder.predict(b_in)).reshape((b_in.shape[0], 147))))
    latent_pred_x_out_pred = np.transpose(inverse @ np.transpose((b_encoder.predict(b_out)).reshape((b_out.shape[0], 147))))
    print(latent_pred_x_in_pred.shape)

    # Eb(b)
    latent_b_in = (b_encoder.predict(b_in)).reshape(b_in.shape[0], 147)
    latent_b_out = (b_encoder.predict(b_out)).reshape(b_out.shape[0], 147)
    print(latent_b_in.shape)

    # M*Ex(xhat)
    latent_pred_b_in = np.transpose(forward @ np.transpose((x_encoder.predict(x_in_pred)).reshape((x_in_pred.shape[0], 147))))
    latent_pred_b_out = np.transpose(forward @ np.transpose((x_encoder.predict(x_out_pred)).reshape((x_out_pred.shape[0], 147))))
    print(latent_pred_b_out.shape)

    # Determine Metrics for In and Out of Distribution Samples
    met_data_ae_in = []
    met_res_est_in = []
    met_reco_ae_in = []
    met_lat_dat_in = []
    met_lat_par_in = []
    met_trueerr_in = []

    met_data_ae_out = []
    met_res_est_out = []
    met_reco_ae_out = []
    met_lat_dat_out = []
    met_lat_par_out = []
    met_trueerr_out = []

    for i in range(x_in.shape[0]):
        met_data_ae_in.append(la.norm(b_in_ae[i, :, :] - b_in[i, :, :])/la.norm(b_in[i, :, :]))
        met_res_est_in.append(la.norm(b_in_pred[i, :, :] - b_in[i, :, :])/la.norm(b_in[i, :, :]))
        met_reco_ae_in.append(la.norm(x_in_pred_ae[i, :, :] - x_in_pred[i, :, :])/la.norm(x_in_pred[i, :, :]))
        met_lat_par_in.append(la.norm(latent_pred_x_in_pred - latent_x_in_pred)/la.norm(latent_x_in_pred[i, :]))
        met_lat_dat_in.append(la.norm(latent_pred_b_in[i, :] - latent_b_in[i, :])/la.norm(latent_b_in[i, :]))
        met_trueerr_in.append(la.norm(x_in_pred[i, :, :] - x_in[i, :, :])/la.norm(x_in[i, :, :]))

    for i in range(x_out.shape[0]):
        met_data_ae_out.append(la.norm(b_out_ae[i, :, :] - b_out[i, :, :])/la.norm(b_out[i, :, :]))
        met_res_est_out.append(la.norm(b_out_pred[i, :, :] - b_out[i, :, :])/la.norm(b_out[i, :, :]))
        met_reco_ae_out.append(la.norm(x_out_pred_ae[i, :, :] - x_out_pred[i, :, :])/la.norm(x_out_pred[i, :, :]))
        met_lat_par_out.append(la.norm(latent_pred_x_out_pred - latent_x_out_pred)/la.norm(latent_x_out_pred[i, :]))
        met_lat_dat_out.append(la.norm(latent_pred_b_out[i, :] - latent_b_out[i, :])/la.norm(latent_b_out[i, :]))
        met_trueerr_out.append(la.norm(x_out_pred[i, :, :] - x_out[i, :, :])/(la.norm(x_out[i, :, :])))

    metrics = [
        (met_data_ae_in, met_data_ae_out, '||Db(Eb(b))-b|| / ||b||'),
        (met_res_est_in, met_res_est_out, '||Db(M*Ex(xhat))-b|| / ||b||'),
        (met_reco_ae_in, met_reco_ae_out, '||Dx(Ex(xhat))-xhat|| / ||xhat||'),
        (met_lat_par_in, met_lat_par_out, '||Mdag*Eb(b)-Ex(xhat)|| / ||Ex(xhat)||'),
        (met_lat_dat_in, met_lat_dat_out, '||M*Ex(xhat)-Eb(b)|| / ||Eb(b)||')
    ]

    # Histograms
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    color_in_dist = '#0072B2'  # Sky blue
    color_out_dist = '#D55E00'  # Orange
    for i, (data_in, data_out, data_label) in enumerate(metrics):
        bins=np.histogram(np.hstack((data_in, data_out)), bins=50)[1]
        axes[i].hist(data_in, bins=bins, alpha=0.75, label='In Distribution',  color=color_in_dist)
        axes[i].hist(data_out, bins=bins, alpha=0.25, label='Out of Distribution', color=color_out_dist)
        axes[i].set_xlabel(data_label)
    axes[4].legend()
        
    plt.suptitle('PAIR Metrics for Out of Distribution Detection')
    plt.tight_layout()
    plt.savefig('images/histo')
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, (data_in, data_out, data_label) in enumerate(metrics):
        axes[i].hist(data_in, bins=50, color='blue')
        axes[i].set_xlabel(data_label)
        
    plt.suptitle('PAIR Metrics for Out of Distribution Detection')
    plt.tight_layout()
    plt.savefig('images/histo_in')
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, (data_in, data_out, data_label) in enumerate(metrics):
        axes[i].hist(data_out, bins=50, color='red')
        axes[i].set_xlabel(data_label)
        
    plt.suptitle('PAIR Metrics for Out of Distribution Detection')
    plt.tight_layout()
    plt.savefig('images/histo_out')
    plt.show()

    for i, (data_in, data_out, data_label) in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(3, 3))
        bins=np.histogram(np.hstack((data_in, data_out)), bins=50)[1]
        ax.hist(data_in, bins=bins, alpha=0.75, label='In Distribution',  color=color_in_dist)
        ax.hist(data_out, bins=bins, alpha=0.25, label='Out of Distribution', color=color_out_dist)
        # ax.set_xlabel(data_label)
        plt.tight_layout()
        if i == 4:
            ax.legend()
        plt.savefig('images/histo'+str(i))
        plt.show()

    # Scatterplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, (data_in, data_out, data_label) in enumerate(metrics):
        axes[i].scatter(data_in, met_trueerr_in, label='In Distribution',  color='blue', s=25, alpha=0.5)
        axes[i].scatter(data_out, met_trueerr_out, label='Out of Distribution', color='red', s=25, alpha=0.5)
        axes[i].set_xlabel(data_label)
        # axes[i].legend()
        if i==0:
            axes[i].set_ylabel('||xhat-x|| / ||x||')
        
    plt.suptitle('PAIR Metrics for Out of Distribution Detection')
    plt.tight_layout()
    plt.savefig('images/scatter')
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, (data_in, data_out, data_label) in enumerate(metrics):
        axes[i].scatter(data_in, met_trueerr_in, color='blue', s=25)
        axes[i].set_xlabel(data_label)
        # axes[i].legend()
        if i==0:
            axes[i].set_ylabel('||xhat-x|| / ||x||')
        
    plt.suptitle('PAIR Metrics for Out of Distribution Detection on ID Samples')
    plt.tight_layout()
    plt.savefig('images/scatter_in')
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, (data_in, data_out, data_label) in enumerate(metrics):
        axes[i].scatter(data_out, met_trueerr_out, color='red', s=25)
        axes[i].set_xlabel(data_label)
        # axes[i].legend()
        if i==0:
            axes[i].set_ylabel('||xhat-x|| / ||x||')
        
    plt.suptitle('PAIR Metrics for Out of Distribution Detection on OOD Samples')
    plt.tight_layout()
    plt.savefig('images/scatter_out')
    plt.show()

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[1].scatter(met_lat_par_in, met_lat_dat_in, label='In Distribution',  color='blue', s=25, alpha=0.5)
    axes[1].scatter(met_lat_par_out, met_lat_dat_out, label='Out of Distribution',  color='red', s=25, alpha=0.5)
    axes[1].set_ylabel('||Mdag*Eb(b)-Ex(xhat)|| / ||Ex(xhat)||')
    axes[1].set_xlabel('||M*Ex(xhat)-Eb(b)|| / ||Eb(b)||')
    plt.suptitle('PAIR Metrics for Out of Distribution Detection')
    plt.tight_layout()
    plt.savefig('images/scatter_2d_latent_spaces')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))  # Single axis instead of an array of axes
    ax.scatter(met_lat_par_in, met_lat_dat_in, label='In Distribution', color=color_in_dist, marker='o', s=25, alpha=0.75, linewidth=0)
    ax.scatter(met_lat_par_out, met_lat_dat_out, label='Out of Distribution', color=color_out_dist, marker='s', s=25, alpha=0.25, linewidth=0)
    # ax.set_ylabel('||Mdag*Eb(b)-Ex(xhat)|| / ||Ex(xhat)||')
    # ax.set_xlabel('||M*Ex(xhat)-Eb(b)|| / ||Eb(b)||')
    ax.legend()  # Adding a legend to differentiate the data sets
    # plt.suptitle('PAIR Metrics for Out of Distribution Detection')
    plt.tight_layout()
    plt.savefig('images/scatter_2d_latent_spaces')
    plt.show()

    # Visualize some reconstructed OOD images
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_out_pred[i*500+1], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('images/reconstructed_notmnist_images.png')
    plt.show()

    print('done')