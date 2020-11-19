from pathlib import Path
from typing import Optional, Tuple

import attr
import humanize
import yaml
from tensorflow.python.keras.utils.layer_utils import count_params
from yaml import YAMLObject

from .data import Volume
from .model import Model
from .volume_img_segm import VolumeImgSegmSequence


@attr.s(auto_attribs=True)
class Metadata(YAMLObject):
    yaml_tag = "!Metadata"

    model_name: str = None

    @attr.s(auto_attribs=True)
    class Paths(YAMLObject):
        yaml_tag = "!Paths"

        model: str = None
        autosave: str = None
        logger: str = None
        architecture_fig: str = None
        summary_txt: str = None
        history_csv: str = None
        metadata_yml: str = None

    paths: Paths = None

    @attr.s(auto_attribs=True)
    class Dataset(YAMLObject):
        yaml_tag = "!Dataset"

        @attr.s(auto_attribs=True)
        class Volume(YAMLObject):
            yaml_tag = "!Volume"

            filename: Optional[str] = None
            shape: Optional[Tuple[int]] = None
            dtype: Optional[Tuple[int]] = None
            mem_size_bytes: int = None
            mem_size_human: str = None

        data: Optional[Volume] = None
        labels: Optional[Volume] = None

        sliced_axes: Tuple[int] = None
        crop_size: int = None

        x_batch_shape: Optional[Tuple[int]] = None
        x_batch_dtype: Optional[str] = None

        y_batch_shape: Optional[Tuple[int]] = None
        y_batch_dtype: Optional[str] = None

    train: Optional[Dataset] = None
    val: Optional[Dataset] = None

    @attr.s(auto_attribs=True)
    class Architecture(YAMLObject):
        yaml_tag = "!Architecture"

        n_params_total: int = None
        n_params_total_human: str = None
        n_params_trainable: int = None
        n_params_trainable_human: str = None
        n_params_nontrainable: int = None
        n_params_nontrainable_human: str = None
        model_generator_function: str = None
        u_net__n_filters_0: int = None
        input_shape: int = None

    architecture: Architecture = None

    batch_size: Optional[int] = None
    n_batches_per_epoch: int = None
    n_examples_per_epoch: int = None
    n_examples_per_epoch_human: str = None
    n_epochs: int = None
    optimizer: str = None
    learning_rate: float = None
    loss_func: str = None

    @classmethod
    def build(
            cls,
            model, 
            model_paths: Model,
            volume_paths: Volume,
            train_generator: VolumeImgSegmSequence,
            val_generator: VolumeImgSegmSequence,
            nb_filters_0, input_shape,
            n_epochs
    ):

        train_x, train_y = train_generator[0]
        val_x, val_y = val_generator[0]
        batch_size = train_generator.batch_size
        n_batches_per_epoch = len(train_generator)
        n_examples_per_epoch = batch_size * n_batches_per_epoch

        # just syntatic sugar
        ds = Metadata.Dataset
        vol = Metadata.Dataset.Volume
        archi = Metadata.Architecture
        pth = Metadata.Paths

        optimizer_class = model.optimizer.__class__
        optimizer_name = f"{optimizer_class.__module__}.{optimizer_class.__name__}"
        learning_rate = float(model.optimizer.lr)

        # todo use keywords everywhere
        return cls(
            model_name=model.name,
            paths=pth(
                str(model_paths.model_path),
                str(model_paths.autosaved_model_path),
                str(model_paths.logger_path),
                str(model_paths.summary_path),
                str(model_paths.history_path),
                metadata_yml=str(model_paths.metadata_yml_path)
            ),
            train=ds(
                # todo make build classmethod for the sub classes as well
                vol(
                    str(volume_paths.train_data_path),
                    str(train_generator.source_volume.shape),
                    train_generator.source_volume.dtype.name,
                    train_generator.source_volume.nbytes,
                    humanize.naturalsize(train_generator.source_volume.nbytes)
                ),
                vol(
                    str(volume_paths.train_labels_path),
                    str(train_generator.label_volume.shape),
                    train_generator.label_volume.dtype.name,
                    train_generator.label_volume.nbytes,
                    humanize.naturalsize(train_generator.label_volume.nbytes)
                ),
                str(train_generator.axes),
                train_generator.crop_size,
                str(train_x.shape), train_x.dtype.name,
                str(train_y.shape), train_y.dtype.name
            ),
            val=ds(
                vol(
                    str(volume_paths.val_data_path),
                    str(val_generator.source_volume.shape),
                    val_generator.source_volume.dtype.name,
                    val_generator.source_volume.nbytes,
                    humanize.naturalsize(val_generator.source_volume.nbytes)
                ),
                vol(
                    str(volume_paths.val_labels_path),
                    str(val_generator.label_volume.shape),
                    val_generator.label_volume.dtype.name,
                    val_generator.label_volume.nbytes,
                    humanize.naturalsize(val_generator.label_volume.nbytes)
                ),
                str(val_generator.axes),
                val_generator.crop_size,
                str(val_x.shape), val_x.dtype.name,
                str(val_y.shape), val_y.dtype.name
            ),
            architecture=archi(
                model.count_params(), humanize.intword(model.count_params()),
                count_params(model.trainable_weights), humanize.intword(count_params(model.trainable_weights)),
                count_params(model.non_trainable_weights), humanize.intword(count_params(model.non_trainable_weights)),
                model.factory_function, nb_filters_0,
                str(input_shape)
            ),
            batch_size=batch_size,
            n_batches_per_epoch=n_batches_per_epoch,
            n_examples_per_epoch=n_examples_per_epoch,
            n_examples_per_epoch_human=humanize.intcomma(n_examples_per_epoch),
            n_epochs=n_epochs,
            optimizer=optimizer_name,
            learning_rate=f"{learning_rate:.2e}",
            loss_func=f"{model.loss.__module__}.{model.loss.__name__}"
        )

    @property
    def yaml_str(self) -> str:
        return yaml.dump(self, default_flow_style=False, indent=4)

    def save_yaml_file(self, metadata_yml_path: Path) -> None:
        with metadata_yml_path.open("w") as f:
            yaml.dump(self, f, default_flow_style=False, indent=4)



