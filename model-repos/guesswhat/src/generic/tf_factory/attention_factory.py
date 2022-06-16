import tensorflow as tf

from neural_toolbox.attention import compute_attention, compute_glimpse, compute_convolution_pooling


def get_attention(feature_map, context, config, is_training, dropout_keep, reuse=False):
    attention_mode = config.get("mode", None)

    if attention_mode == "none":
        image_out = feature_map

    elif attention_mode == "max":
        image_out = tf.reduce_max(feature_map, axis=(1, 2))

    elif attention_mode == "mean":
        image_out = tf.reduce_mean(feature_map, axis=(1, 2))

    elif attention_mode == "classic":
        image_out = compute_attention(feature_map,
                                      context,
                                      no_mlp_units=config['no_attention_mlp'],
                                      fuse_mode=config['fuse_mode'],
                                      keep_dropout=dropout_keep,
                                      reuse=reuse)

    elif attention_mode == "glimpse":
        image_out = compute_glimpse(feature_map,
                                    context,
                                    no_glimpse=config['no_glimpses'],
                                    glimpse_embedding_size=config['no_attention_mlp'],
                                    keep_dropout=dropout_keep,
                                    reuse=reuse)

    elif attention_mode == "conv_pooling":
        image_out = compute_convolution_pooling(feature_map,
                                                no_mlp_units=config['no_attention_mlp'],
                                                is_training=is_training,
                                                reuse=reuse)

    else:
        assert False, "Wrong attention mode: {}".format(attention_mode)

    return image_out
