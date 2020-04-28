import tensorflow as tf


class YoloV1Loss(tf.keras.losses.Loss):
    def __init__(self, params):
        """
        YOLO v1 loss
        Arguments:
            S - int / 이미지 분할 구역 수 (이미지를 S x S 구역으로 분할)
            B - int / 구역마다 predictor 수
            lambda_coord - float / localization loss 가중치
            lambda_noobj - float / object loss 중 오브젝트가 없는 구역에 대한 가중치
            name - str / 이름
        """
        super(YoloV1Loss, self).__init__(reduction=tf.keras.losses.Reduction.AUTO)

        self.S = params.network.yolo.S
        self.B = params.network.yolo.B
        self.classes = params.network.yolo.classes
        self.lambda_coord = params.network.yolo.lambda_coord
        self.lambda_noobj = params.network.yolo.lambda_noobj

    @tf.function
    def call(self, true, pred):
        true_bndbox, pred_bndbox = true[..., :5], pred[..., :self.B * 5]
        true_bndbox = tf.reshape(true_bndbox, (-1, self.S, self.S, 1, 5))
        pred_bndbox = tf.reshape(pred_bndbox, (-1, self.S, self.S, self.B, 5))

        # iou
        iou = self.IOU(true_bndbox, pred_bndbox, self.S, self.B)  # n, S, S, B
        iou_onehot = tf.one_hot(tf.argmax(iou, axis=-1), depth=self.B, axis=-1)  # n, S, S, B

        # responsibility
        obj_1 = true_bndbox[..., 4]  # n, S, S, 1
        obj_1_ij = obj_1 * iou_onehot  # n, S, S, B
        noobj_1_ij = 1. - obj_1  # n, S, S, 1

        # bndbox loss
        xy, xy_pred = true_bndbox[..., :2], pred_bndbox[..., :2]
        xy_loss = tf.reduce_sum(obj_1_ij[..., None] * tf.square(xy - xy_pred), axis=[1, 2, 3, 4])
        wh, wh_pred = true_bndbox[..., 2:4], pred_bndbox[..., 2:4]
        wh_loss = tf.reduce_sum(obj_1_ij[..., None] * tf.square(wh - wh_pred), axis=[1, 2, 3, 4])

        # objectness loss
        C, C_pred = iou, pred_bndbox[..., -1]
        # C_obj_loss = tf.reduce_sum(obj_1_ij * tf.square(C - C_pred), axis=[1, 2, 3])
        C_obj_loss = tf.reduce_sum(obj_1_ij * tf.square(1. - C_pred), axis=[1, 2, 3])
        C_noobj_loss = tf.reduce_sum(noobj_1_ij * tf.square(C_pred), axis=[1, 2, 3])

        loss = self.lambda_coord * (xy_loss+wh_loss) + C_obj_loss + self.lambda_noobj * C_noobj_loss
        return tf.reduce_mean(loss)

    def IOU(self, bndbox_0, bndbox_1, S, B):
        #  bndbox : n, S, S, 1, 5 / n, S, S, B, 5
        bndbox_0 = tf.tile(bndbox_0, (1, 1, 1, B, 1))
        bndbox = tf.stack([bndbox_0, bndbox_1], axis=-1)  # n, S, S, B, 5, 2

        xy = bndbox[..., :2, :]
        wh = tf.square(bndbox[..., 2:4, :])

        left_top = tf.reduce_max(xy / S - wh / 2, axis=-1)  # n, S, S, B, 2
        right_bot = tf.reduce_min(xy / S + wh / 2, axis=-1)
        small_wh = tf.reduce_min(wh, axis=-1)

        intersection_wh = tf.clip_by_value(right_bot - left_top, 0., small_wh)  # n, S, S, B, 2
        intersection = tf.reduce_prod(intersection_wh, axis=-1)  # n, S, S, B

        union = tf.reduce_sum(tf.reduce_prod(wh, axis=-2), axis=-1) - intersection

        return intersection / (union + 1e-4)

