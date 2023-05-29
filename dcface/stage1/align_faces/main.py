from face_detector import FaceDetector
from similarity_transform.align_trans import warp_and_crop_face, reference_landmark
import argparse
from tqdm import tqdm
import pandas as pd
import os
import re


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_all_files(root, extension_list=['.png', '.jpg', '.jpeg'], sort=False):

    all_files = list()
    for (dirpath, dirnames, filenames) in os.walk(root):
        all_files += [os.path.join(dirpath, file) for file in filenames]
    if extension_list is None:
        return all_files
    all_files = list(filter(lambda x: os.path.splitext(x)[1] in extension_list, all_files))
    if sort:
        all_files = natural_sort(all_files)
    return all_files


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../unconditional_generation/unconditional_samples')
    parser.add_argument('--save_root', type=str, default='../unconditional_generation/unconditional_samples_aligned')
    parser.add_argument('--fallback_method', type=str, default='pass')
    parser.add_argument('--detected_shape', type=int, default=112)
    parser.add_argument('--aligned_shape', type=int, default=112)
    parser.add_argument('--pad_ratio', type=float, default=0.05)
    args = parser.parse_args()

    if args.root.endswith('/'):
        args.root = args.root[:-1]
    if args.save_root.endswith('/'):
        args.save_root = args.save_root[:-1]

    det_size = (args.detected_shape, args.detected_shape)
    align_size = (args.aligned_shape, args.aligned_shape)
    print(f'saving at {args.save_root}')

    detector = FaceDetector(output_shape=det_size,
                            square_bbox=True,
                            fallback=args.fallback_method)

    image_paths = get_all_files(args.root, extension_list=['.png', '.jpg', '.jpeg'])

    success_result = []
    fail_result = []
    for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        success, orig_img, cropped, bbox, crop_landmark, score = detector.detect(image_path)
        if success:
            aligned_img, cv_tfm = warp_and_crop_face(cropped, crop_landmark, reference_landmark(), crop_size=align_size)
            row = {'status': 'success', 'image_path':image_path, 'idx':idx}
            success_result.append(row)

            # aligned save
            save_path = image_path.replace(args.root, args.save_root)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            aligned_img.save(save_path)
        else:
            row = {'status': 'fail', 'image_path':image_path, 'idx':idx}
            fail_result.append(row)

    success_result = pd.DataFrame(success_result)
    success_result.to_csv(os.path.join(args.save_root, 'success.csv'))
    fail_result = pd.DataFrame(fail_result)
    fail_result.to_csv(os.path.join(args.save_root, 'fail.csv'))
