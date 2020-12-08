"""Obtain the line images from the Bangale Writing data."""
import argparse
import glob
import json
import os

import cv2
import exputils
import numpy as np


def script_args(parser):
    group = parser.add_argument_group('line_imgs', 'Line image extraction.')

    group.add_argument(
        'input_dir',
        help='filepath to a directory of input files.',
    )

    group.add_argument(
        'output_dir',
        default=None,
        help='Output directory',
    )


def check_points_format(points):
    return points[0][0] < points[1][0] and points[0][1] < points[1][1]


def get_line_imgs(img_path, json_path):
    """Loads the MEVM with the given state and a NominalEncoder with the given
    json and combines them into the MEVM wrapper class and saves the state.
    """
    # Load image and JSON
    img = cv2.imread(img_path)
    with open(json_path, 'r') as openf:
        img_dict = json.load(openf)

    # Use the word bounding boxes to find the lines: Loop through the words,
    # assumed to be in order from left-to-right and once the next bounding box
    # has a minimum x less than the minimum x of the current bbox, then start
    # new line.
    line_idx = []
    start_idx = 0
    next_i = 0
    if not check_points_format(img_dict['shapes'][0]['points']):
        raise ValueError(
            f'The JSON format for points is incorrect for `{json_path}`'
        )

    for next_i, shape in enumerate(img_dict['shapes'][1:], 1):
        if not check_points_format(shape['points']):
            raise ValueError(
                f'The JSON format for points is incorrect for `{json_path}`'
            )

        if (
            shape['points'][0][0]
            < img_dict['shapes'][next_i - 1]['points'][0][0]
        ):
            # If current min x is less than prior min x, save slice, start new
            line_idx.append((start_idx, int(next_i)))
            start_idx = int(next_i)

    # The last line of words still has yet to be added, so add it
    line_idx.append((start_idx, int(next_i) + 1))

    # After determining which word bounding boxes belong to which line,
    # reorganize the JSON to include this line information, saving line bbox
    lines = [img_dict['shapes'][start:end] for start, end in line_idx]

    if len(line_idx) <= 1:
        raise ValueError('jimminy cricket!')

    # Crop the image into these separate line images based on lines' bboxes
    line_imgs = []
    for line in lines:
        # Get min and max x and y across all bounding boxes in the line
        min_x = np.inf
        min_y = np.inf
        max_x = 0
        max_y = 0

        for bbox in line:
            min_x = min(min_x, int(bbox['points'][0][0]))
            min_y = min(min_y, int(bbox['points'][0][1]))
            max_x = max(max_x, int(bbox['points'][1][0]))
            max_y = max(max_y, int(bbox['points'][1][1]))

        # Crop the line image from img
        line_imgs.append(img[min_y:max_y, min_x:max_x])

    img_dict['shapes'] = lines

    return line_imgs, img_dict


def multiple_get_line_imgs(input_dir, output_dir, img_ext='jpg'):
    output_dir = exputils.io.create_dirs(output_dir)

    for path in glob.iglob(os.path.join(input_dir, f'*.{img_ext}')):
        filepath, ext = os.path.splitext(path)
        json_path = f'{filepath}.json'

        if not os.path.isfile(json_path):
            raise IOError(f'Missing JSON file: {json_path}')

        # Get line images
        line_imgs, line_dict = get_line_imgs(path, json_path)
        basename = os.path.basename(filepath)

        out_path = exputils.io.create_dirs(os.path.join(output_dir, basename))
        out_base = os.path.join(out_path, basename)

        # Save the resulting images in the new directory w/ ordered numbers
        for i, img in enumerate(line_imgs):
            cv2.imwrite(f'{out_base}-{i}.jpg', img)

        # Save their JSON info, expected as a single JSON for set of lines
        with open(f'{out_base}.json', 'w') as openf:
            json.dump(line_dict, openf, indent=4, sort_keys=True)


if __name__ == '__main__':
    args = exputils.io.parse_args(
        'Obtains line images from each image, saving the json info with them.',
        custom_args=script_args,
    )

    if os.path.isdir(args.input_dir):
        multiple_get_line_imgs(args.input_dir, args.output_dir)
    else:
        raise ValueError(' '.join([
            'Handles a directory of jsons and jpg images who share the same',
            'filename, minus the extention.',
        ]))
