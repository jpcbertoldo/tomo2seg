import numpy as np


hist_bins = np.linspace(0, 256, 257).astype(int)

data_hist, _ = np.histogram(
    data_volume.ravel(),
    bins=hist_bins,
    density=False,
)

labels_volume_raveled_t = tf.convert_to_tensor(
    labels_volume.ravel(), dtype=tf.int8
)

data_hists_per_label = []
data_hists_per_label_global_prop = []
n_voxels = np.sum(labels_counts)

for label_idx in labels_idx:
    logger.debug(f"Computing histogram for {label_idx=}")

    label_data_hist_t = tf.histogram_fixed_width(
        values=data_volume_raveled_t[labels_volume_raveled_t == label_idx],
        value_range=tf.constant(hist_range, dtype=data_volume_raveled_t.dtype),
        nbins=n_bins,
        dtype=data_volume_raveled_t.dtype,
        name=f"{volume.fullname}.data-histogram"
    )
    data_hists_per_label.append(
        (label_data_hist_t / tf.math.reduce_sum(label_data_hist_t)).numpy().tolist()
    )
    data_hists_per_label_global_prop.append(
        (label_data_hist_t.numpy() / n_voxels).tolist()
    )


bins = np.linspace(*hist_range, n_bins).astype(int).tolist()
