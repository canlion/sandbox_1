import numpy as np
import tensorflow as tf
import imgaug
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox
from imgaug.augmentables.batches import UnnormalizedBatch


class YoloData:
    def __init__(self, params):
        self.S = params.network.yolo.S
        self.B = params.network.yolo.B
        self.classes = params.network.yolo.classes
        self.img_size = params.network.input_size
        self.map_fn_train = None
        self.map_fn_eval = None
        self.aug_seq = imgaug.augmenters.Sequential([
            imgaug.augmenters.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5), mul_saturation=(0.5, 1.5)),
            imgaug.augmenters.MultiplyBrightness(mul=(0.8, 1.2)),
            imgaug.augmenters.contrast.GammaContrast(gamma=(0.7, 1.3)),
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.Affine(rotate=(-3, 3), shear=(-3, 3)),
            imgaug.augmenters.CropAndPad(percent=(-.1, .1)),
            imgaug.augmenters.GaussianBlur(sigma=(0.0, 4.0))
        ])
        self.map_fn()

    def yolo_label(self, bndboxes, labels):
        # bndboxes = list((l, t, r, b), ...)
        grid = np.zeros((self.S, self.S, 5 + self.classes))
        for bndbox, label in zip(bndboxes, labels):
            bndbox = tf.clip_by_value(bndbox, 1e-4, 1 - 1e-4)
            center_x, center_y = (bndbox[:2] + bndbox[2:]) / 2
            w, h = bndbox[2:] - bndbox[:2]

            grid_x, x = np.divmod(center_x, 1/self.S)
            grid_y, y = np.divmod(center_y, 1/self.S)

            grid_x, grid_y = int(grid_x), int(grid_y)

            grid[grid_y, grid_x] = 0.
            grid[grid_y, grid_x, 0] = x * self.S
            grid[grid_y, grid_x, 1] = y * self.S
            grid[grid_y, grid_x, 2] = np.sqrt(w)
            grid[grid_y, grid_x, 3] = np.sqrt(h)
            grid[grid_y, grid_x, 4] = 1.
            grid[grid_y, grid_x, label-self.classes] = 1.

        grid = np.concatenate([grid[..., :5], grid], axis=-1)

        return grid

    def augmentation(self, image, bndboxes):
        H, W, C = image.shape
        bndboxes = BoundingBoxesOnImage(
            [BoundingBox(x1=l*W, x2=r*W, y1=t*H, y2=b*H) for l, t, r, b in bndboxes],
            shape=image.shape
        )

        img_aug, bndboxes_aug = self.aug_seq(image=image, bounding_boxes=bndboxes)
        bndboxes_aug = bndboxes_aug.to_xyxy_array() / np.array([W, H, W, H])

        return img_aug, bndboxes_aug

    def yolo_data(self, image, labels, xmin, ymin, xmax, ymax, augmentation=True):
        bndboxes = tf.transpose(tf.stack([xmin, ymin, xmax, ymax], axis=0)).numpy()
        image = image.numpy()

        if augmentation:
            image, bndboxes = self.augmentation(image, bndboxes)

        image = tf.image.resize(image, self.img_size[:2])
        image = tf.cast(image, tf.float32)
        y = self.yolo_label(bndboxes, labels)

        return image, y

    def map_fn(self):
        self.map_fn_train = lambda x: tf.py_function(self.yolo_data,
                                                    [x['image'], x['label'], x['xmin'],
                                                     x['ymin'], x['xmax'], x['ymax'], True],
                                                    [tf.float32, tf.float32])
        self.map_fn_eval = lambda x: tf.py_function(self.yolo_data,
                                                   [x['image'], x['label'], x['xmin'],
                                                    x['ymin'], x['xmax'], x['ymax'], False],
                                                   [tf.float32, tf.float32])


    def train_batch_generator(self, ds, batch_size, core=8):
        def map_fn(x):
            return x['image'], x['label'], [x['xmin'], x['ymin'], x['xmax'], x['ymax']]

        def generator(ds):
            def multicore_augmentation(img_list, coord_list):
                # coord_list = [np.transpose(np.stack(coord, axis=0)) for coord in coord_list]
                coord_list = [np.transpose(coord) for coord in coord_list]
                bndbox_list = list()
                for img, coord in zip(img_list, coord_list):
                    H, W, C = img.shape
                    bndbox_list.append(
                        BoundingBoxesOnImage(
                            [BoundingBox(x1=l * W, x2=r * W, y1=t * H, y2=b * H) for l, t, r, b in coord],
                            shape=img.shape
                        )
                    )

                batch_per_core = batch_size//core
                batches = [
                    UnnormalizedBatch(images=img_list[i:i+batch_per_core],
                                      bounding_boxes=bndbox_list[i:i+batch_per_core])
                    for i in range(0, batch_size, batch_per_core)
                ]

                batches_aug = list(self.aug_seq.augment_batches(batches, background=True))
                return batches_aug

            def make_label(aug_result, label_list):
                img_list, bndbox_list = list(), list()
                for batches in aug_result:
                    for img, bndbox in zip(batches.images_aug, batches.bounding_boxes_aug):
                        H, W, C = img.shape
                        img_list.append(tf.image.resize(img, self.img_size[:2]))
                        bndbox = bndbox.to_xyxy_array() / np.array([W, H, W, H])
                        bndbox_list.append(bndbox)

                grids = list()
                for label, bndbox in zip(label_list, bndbox_list):
                    grids.append(self.yolo_label(bndbox, label))

                return np.stack(img_list, axis=0), np.stack(grids, axis=0)


            img_list, label_list, coord_list = list(), list(), list()
            for img, label, coord in ds:
                if len(img_list) > batch_size:
                    img_list.clear(); label_list.clear(); coord_list.clear()
                img_list.append(np.array(img))
                label_list.append(np.array(label))
                coord_list.append(np.array(coord))
                if len(img_list) == batch_size:
                    aug_result = multicore_augmentation(img_list, coord_list)
                    x, y = make_label(aug_result, label_list)
                    yield x, y

        ds = ds.map(map_fn, tf.data.experimental.AUTOTUNE)
        return generator(ds)



