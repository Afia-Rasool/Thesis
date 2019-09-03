from .builder import build_fpn_ASPP
from ..backbones import get_backbone, get_feature_layers
from ..common import freeze_model
from ..common import legacy_support

old_args_map = {
    'freeze_encoder': 'encoder_freeze',
    'fpn_layers': 'encoder_features',
    'use_batchnorm': 'pyramid_use_batchnorm',
    'dropout': 'pyramid_dropout',
    'interpolation': 'final_interpolation',
    'upsample_rates': None,  # removed
    'last_upsample': None,  # removed
    'input_tensor': None, # removed
}


@legacy_support(old_args_map)
def FPN_ASPP(backbone_name=None,
        input_shape=(None, None, 3),
        classes=None,
        activation='softmax',
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        pyramid_block_filters=256,
        pyramid_use_batchnorm=True,
        pyramid_dropout=None,
        final_interpolation='bilinear',
        **kwargs):

    backbone = get_backbone(backbone_name,
                            input_shape=input_shape,
                            weights=encoder_weights,
                            include_top=False)

    if encoder_features == 'default':
        encoder_features = get_feature_layers(backbone_name, n=3)

    upsample_rates = [2] * len(encoder_features)

    model = build_fpn_ASPP(backbone, encoder_features,
                      input_shape=(224, 224, 3),
                      classes=classes,
                      pyramid_filters=pyramid_block_filters,
                      segmentation_filters=pyramid_block_filters // 2,
                      upsample_rates=upsample_rates,
                      use_batchnorm=pyramid_use_batchnorm,
                      dropout=pyramid_dropout,
                      interpolation=final_interpolation,
                      activation=activation)

    if encoder_freeze:
        freeze_model(backbone)

    model.name = 'fpn-{}'.format(backbone.name)

    return model
