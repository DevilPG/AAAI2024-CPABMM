import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
import numpy as np
import imageio
from skimage import img_as_ubyte


def reconstruction(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    mp4_dir = os.path.join(log_dir, 'reconstruction/mp4')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, kp_detector=kp_detector,
                         bg_predictor=bg_predictor, dense_motion_network=dense_motion_network)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    if not os.path.exists(mp4_dir):
        os.makedirs(mp4_dir)
    
    loss_list = []

    inpainting_network.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    if bg_predictor:
        bg_predictor.eval()

    for it, x in tqdm(enumerate(dataloader)):
        # with torch.no_grad():
        predictions = []
        visualizations = []
        if torch.cuda.is_available():
            x['video'] = x['video'].cuda()
        kp_source = kp_detector(x['video'][:, :, 0])
        for frame_idx in range(x['video'].shape[2]):
            source = x['video'][:, :, 0]
            driving = x['video'][:, :, frame_idx]
            kp_driving = kp_detector(driving)
            bg_params = None
            if bg_predictor:
                bg_params = bg_predictor(source, driving)
            
            # dense_motion = dense_motion_network(source_image=source, kp_driving=kp_driving,
            #                                     kp_source=kp_source, bg_param = bg_params, 
            #                                     dropout_flag = False, retain_graph = False)
            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_driving,
                                                kp_source=kp_source, bg_param = bg_params, 
                                                dropout_flag = False)
            out = inpainting_network(source, dense_motion)
            out['kp_source'] = kp_source
            out['kp_driving'] = kp_driving

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

            visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                driving=driving, out=out)
            visualizations.append(visualization)
            loss = torch.abs(out['prediction'] - driving).mean().cpu().detach().numpy()
            
            loss_list.append(loss)
        # print(np.mean(loss_list))
        try:
            imageio.mimsave(os.path.join(mp4_dir, x['name'][0] + '.mp4'), [img_as_ubyte(frame) for frame in predictions], fps=15)
        except:
            # print(loss_list)
            # for frame in predictions:
            #     # print()
            #     print(frame.min(), frame.max())
            imageio.mimsave(os.path.join(mp4_dir, x['name'][0] + '.mp4'), [img_as_ubyte(np.clip(frame,0,1)) for frame in predictions], fps=15)
            print(x['name'])
        # predictions = np.concatenate(predictions, axis=1)
        # imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

    print("Reconstruction loss: %s" % np.mean(loss_list))
