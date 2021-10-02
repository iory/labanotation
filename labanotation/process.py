from pathlib import Path

import cv2
from eos import makedirs
import numpy as np
from pybsc import nsplit
from pybsc import save_json
from pybsc.video_utils import extract_target_frame_from_timestamp

from labanotation.io import read_from_labanotation_suite
from labanotation.labanotation_utils import arrange_labanotations
from labanotation.labanotation_utils import calculate_unfiltered_labanotations
from labanotation.labanotation_utils import calculate_z_axis
from labanotation.labanotation_utils import extract_target_labanotations
from labanotation import parallel_energy
from labanotation import total_energy
from labanotation.visualization import labanotation_to_image


def get_labanotation_results(csv_filepath,
                             output_path,
                             video_path=None,
                             gauss_window_size=61,
                             gauss_sigma=5,
                             base_rotation_style='update',
                             save_laban_image=True):
    output_path = Path(output_path)
    csv_filepath = Path(csv_filepath)

    joint_positions_dict, timestamps = read_from_labanotation_suite(
        csv_filepath)

    z_axis = calculate_z_axis(joint_positions_dict['shoulder_left'],
                              joint_positions_dict['shoulder_right'],
                              joint_positions_dict['spine_navel'],)
    lines = calculate_unfiltered_labanotations(
        joint_positions_dict,
        base_rotation_style=base_rotation_style,
        z_axis=z_axis)
    labanotations_list = [line[1] for line in lines]
    positions_list = [line[0] for line in lines]
    keyframe_indices, valley_output, energy_list, naive_energy_list = \
        parallel_energy(
            labanotations_list,
            positions_list,
            gauss_window_size=gauss_window_size,
            gauss_large_sigma=gauss_sigma)
    all_labans = arrange_labanotations(labanotations_list)
    keyframe_indices, labans = extract_target_labanotations(
        keyframe_indices, all_labans)

    frame_to_second = (timestamps[-1] - timestamps[0]) / len(all_labans)
    keyframe_timestamps = frame_to_second * keyframe_indices

    if save_laban_image is True:
        makedirs(output_path)
        sp = max(1, len(keyframe_timestamps) // 20)
        for i, (a, b, idx) in enumerate(
                zip(nsplit(keyframe_timestamps, sp), nsplit(labans, sp),
                    nsplit(keyframe_indices, sp))):
            fm_a = np.array(a) - a[0]
            if video_path is not None:
                new_out = output_path \
                    / 'labanotation-{0:06}'.format(i)
                makedirs(new_out)
                imgs = []
                for j, c in enumerate(a):
                    img = extract_target_frame_from_timestamp(
                        video_path, c)
                    imgs.append(img)
                    if img is not None:
                        cv2.imwrite(str(new_out / '{0:06}.jpg'.format(j)), img)
                laban_img = labanotation_to_image(
                    fm_a, b, scale=100, imgs=imgs)
            else:
                laban_img = labanotation_to_image(fm_a, b, scale=100)
            cv2.imwrite(
                str(output_path
                    / 'labanotation-{0:06}.png'.format(i)),
                cv2.rotate(laban_img, cv2.ROTATE_90_CLOCKWISE))

    wrist_right_keyframe_indices, right_energy, _, _ = total_energy(
        timestamps,
        joint_positions_dict['wrist_right'],
        gauss_window_size=gauss_window_size,
        gauss_sigma=gauss_sigma)
    wrist_left_keyframe_indices, left_energy, _, _ = total_energy(
        timestamps,
        joint_positions_dict['wrist_left'],
        gauss_window_size=gauss_window_size,
        gauss_sigma=gauss_sigma)

    laban_data = {'keyframe_timestamps': keyframe_timestamps.tolist(),
                  'keyframe_labanotations': labans,
                  'keyframe_indices': keyframe_indices.tolist(),
                  'timestamps': timestamps.tolist(),
                  'labanotations': all_labans,
                  'wrist_right_keyframe_timestamps':
                  (frame_to_second * wrist_right_keyframe_indices).tolist(),
                  'wrist_right_keyframe_indices':
                  wrist_right_keyframe_indices.tolist(),
                  'right_energy': right_energy.tolist(),
                  'wrist_left_keyframe_timestamps':
                  (frame_to_second * wrist_left_keyframe_indices).tolist(),
                  'wrist_left_keyframe_indices':
                  wrist_left_keyframe_indices.tolist(),
                  'left_energy': left_energy.tolist(),
                  'parameters': {'gauss_window_size': gauss_window_size,
                                 'gauss_sigma': gauss_sigma,
                                 'base_rotation_style': base_rotation_style,
                                 'frame_to_second': frame_to_second}}
    save_json(laban_data, output_path / 'labanotation.json')
