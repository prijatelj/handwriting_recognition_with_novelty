"""Obtain the line images from the Bangale Writing data."""
import argparse
import glob
import json
import os

import cv2
import exputils


def script_args(parser):
    group = parser.add_argument_group('line_imgs', 'Line image extraction.')

    group.add_argument(
        'input_dir',
        help='filepath to a directory of input files.',
        dest='line_imgs',
    )

    group.add_argument(
        '-o',
        '--output_dir',
        default=None,
        help='Output directory',
        dest='line_imgs',
    )


def get_line_imgs(img_path, json_path):
    """Loads the MEVM with the given state and a NominalEncoder with the given
    json and combines them into the MEVM wrapper class and saves the state.
    """
    # Load image and JSON
    img = cv2.imread(img_path)
    with open(json_path, 'r') as openf:
        img_dict = json.load(openf)

    # TODO Get number of lines, use word bboxes to get lines' bboxes.


    # TODO After determining which word bounding boxes belong to which line,
    # reorganize the JSON to include this line information, saving line bbox

    # TODO Cut the image into these separate line images based on lines' bboxes


    return line_imgs, line_img_info


def multiple_get_line_imgs(input_dir, output_dir, img_ext='jpg'):
    output_dir = exputils.io.create_dirs(output_dir)

    for path in glob.iglob(os.path.join(input_dir, f'*.{img_ext}')):
        filepath, ext = os.path.splitext(path)
        json_path = os.path.join(filepath, '.json')

        if not os.path.isfile(json_path):
            raise IOError(f'Missing JSON file: {json_path}')

        # Get line images
        line_imgs, line_img_info = get_line_imgs(path, json_path)
        basename = os.path.basename(filepath)

        out_path = exputils.io.create_dirs(os.path.join(output_dir, basename))
        out_base = os.path.join(out_path, basename)

        # Save the resulting images in the new directory w/ ordered numbers
        for i, img in enumerate(line_imgs):
            cv2.imwrite(f'{out_base}-{i}.jpg')

        # Save their JSON info, expected as a single JSON for set of lines
        with open(f'{out_base}.json','w') as openf:
            json.dump(line_img_info, openf)


if __name__ == '__main__':
    args = exputils.io.parse_args(
        'Obtains the line images from each image, saving the json info with them.',
        custom_args=script_args,
    )

    if os.path.isdir(args.mevm_state_path):
        multiple_get_line_imgs(**vars(args.line_imgs))
    else:
        raise ValueError(' '.join([
            'Handles a directory of jsons and jpg images who share the same',
            'filename, minus the extention.',
        ]))
