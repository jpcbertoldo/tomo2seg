# tomo2seg

## Installation

### Conda environment

Use `anaconda/conda3.7`.

Clone the environ the file `reqs/condaor `reqs/conda-env-hist.yml`.

Then *add this package locally* to your envconda`nt: conda develop .` ( with the terminal in the root of the project).

[todo] test the env reproduction and add comm lines here.

### Data

everything is ignored

## Folders

- `tomo2seg`: contains useful functions, variables, and classes that I use in my notebooks or scripts.

- `nbs`: jupyter notebooks; follow the order of the names to see my development process.

- `scripts`

- `jobs`

- `reqs`

Have fun (:


## GPUs at the Centre des Materiaux

I had issues using them with tensorflow first. Read [this doc](https://docs.google.com/document/d/10OktY72PNIowjBaNCPHcV-zyRyRYhGs7BUjEZalRPKA/edit?usp=sharing) to see how I debugged it and how you should install tensorflow-gpu.

Key specs about the GPUS (2x NVIDIA Quadro P4000:

- gpu model: GP104GL [Quadro P4000]
- cuda  version: 10.1.168
- driver version: 450.57

[Link for more specifications in my debug doc](https://docs.google.com/document/d/10OktY72PNIowjBaNCPHcV-zyRyRYhGs7BUjEZalRPKA/edit#bookmark=id.jj7oewgniyhv).


# TODOs

this is a living list of things that i realize that i can improve during development

[general]
implement pixel-wise weighted jaccard2 loss

[data-management]
todo2: v2 of the sequence generator to get the entire area (or not)
dtype = "uint8"  # todo remove the cast inside the generator?

[metadata]
mirrored_strategy = tf.distribute.MirroredStrategy()  # todo add strategy to metadata
batch per replica
str(model_paths.autosaved_model_path) + ".hdf5",  # todo move this to inside ModelPaths
LR schedule + to the history

[analysis]
add variation_of_information to my metrics notebook   [link] function skimage.metrics.variation_of_information() on skimage

[viz]

[training]
make the history object continue where it stopped
    find the epoch that the training stopped at

[to-cleanup]
todo2: v2 of the sequence generator to get the entire area with random crops
    - and probability of rejecting a crop gven by a function
todo3: predict the whole validation volume
    - remember to declar the aggregation strategy somewhere
todo4: measure some metrics with it
    - volume fraction estimation
    - error blobs
        - FP
        - FN
        - both
        - blob stats
            - min area
            - max area
            - count
            - box dimensions
                - longest length
                - cubic direction
        note:
    - batch size
    - image size
    - n channels
- n crops

todo !!!!!!!!!!!!!!!!!!!!!!!!!!
also save crop predictions so that i can compare models that might have different cropping schemes

# todo add links to the files to notebooks to inspect results?

# todo plot image samples in training

validation_steps=100,  # todo put in summary

# todo add metrics
# todo add wandb?
# todo add analysis in the end, see examples of classif
# todo add callbacks that generate classif examples
# todo print line that I can cccv on the experiments spreadsheet
# todo save the yaml file
# todo (later): separate the analysis part in a separate script so it
#  can be called at any time with another model
# todo experiment with CentralStorageStrategy?

todo notify me if the training breaks or finishes
    epochs=30,  # todo put this in variable
    callbacks=cb,  # todo mention in summary...
        use_multiprocessing=False,   # todo add to summary


    data_hists_per_label_global_prop.append(
        # todo correct this to use shape
        label_data_hist_t.numpy() / 500**3
    )

todo do something to materialize or document the bins used in the histograms

todo compute using keras background so i can put this in the model
todo make option to save crop predictions so i can compare models without the aggregation method


improve my eval notebook
improve the ‘notable slices” logic to spit out per-slice values → use 2d metrics and plot them along the slices to find abnormal slices
automatically spit out the notable slices
compile in a notebook
images of crack
images of armand’s volumes
go back to the fracture volume (jordan’s)
implement weighted loss to force learning the fracture
retrain composite 3d
change the loss function [github] bermanmaxim/LovaszSoftmax
crack with pre-training
modify my training nb to script and hyper optimize (see keras-tuner):
generalize more things in the modular structure
batchnorm: add option for layernorm
generalize the number of convolutions in the block
combine xception-like stuff
combine densenet-like stuff
forward input to every block + last layer
morpholayers
3d
2d
later before classes in march
intermediate images inside the model (for the classes in march)
training with vgg16
later
intermediate segmentations during the training ⇒ make it during a training with the train and val
investigate what happens with the parallel encdec
[todo] write my analysis of jaccard2 properly → [link] jaccard2 loss analysis sketch
re-install tensorflow once the cudnn is installed ⇒ go to go
TO TRIAGEa
add training time to the metadata
automate model comparison (scatter, bar plots) → include nb. params, training time, model size (mb)
