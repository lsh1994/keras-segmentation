import glob
import numpy as np
import cv2
import random
from skimage import color, exposure


def imageSegmentationGenerator(images_path, segs_path, n_classes):

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = sorted(
        glob.glob(
            images_path +
            "*.jpg") +
        glob.glob(
            images_path +
            "*.png") +
        glob.glob(
            images_path +
            "*.jpeg"))
    segmentations = sorted(glob.glob(
        segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg"))

    colors = [
        (random.randint(
            0, 255), random.randint(
            0, 255), random.randint(
                0, 255)) for _ in range(n_classes)]

    assert len(images) == len(segmentations)

    for im_fn, seg_fn in zip(images, segmentations):

        img = cv2.imread(im_fn)
        seg = cv2.imread(seg_fn)
        print(np.unique(seg))

        seg_img = np.zeros_like(seg)

        for c in range(n_classes):
            seg_img[:, :, 0] += ((seg[:, :, 0] == c) *
                                 (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((seg[:, :, 0] == c) *
                                 (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((seg[:, :, 0] == c) *
                                 (colors[c][2])).astype('uint8')

        eqaimg = color.rgb2hsv(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        eqaimg[:, :, 2] = exposure.equalize_hist(eqaimg[:, :, 2])
        eqaimg = color.hsv2rgb(eqaimg)

        cv2.imshow("img", img)
        cv2.imshow("seg_img", seg_img)
        cv2.imshow(
            "equalize_hist_img",
            cv2.cvtColor(
                (eqaimg *
                 255.).astype(
                    np.uint8),
                cv2.COLOR_RGB2BGR))
        cv2.waitKey()


images = "data/dataset1/images_prepped_train/"
annotations = "data/dataset1/annotations_prepped_train/"
n_classes = 11

imageSegmentationGenerator(images, annotations, n_classes)
