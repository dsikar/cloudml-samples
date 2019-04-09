# Quickstart
Requirements:
* Goggle Cloud account with billing enabled
* One registered project
* Dataflow, Computed Engine and Machine Learning Engine APIs enabled
* One bucket named after your project
You this is not the case, best to continue reading. Otherwise, go to Google Cloud Console, open a cloud shell and type:
```
$ git clone https://github.com/dsikar/cloudml-samples.git
$ cd cloudml-samples/coastline
$ . samples.sh
```

# Overview
This code uses the [Coastline](https://codelabs.developers.google.com/codelabs/scd-coastline/index.html?index=..%2F..cloud-quest-scientific-data#0) dataset and is based on the [Flowers](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/flowers) example i.e. implements image-based transfer learning on Cloud ML
In this tutorial you will walk through and results you will monitor consist of four parts: data preprocessing, model training with the transformed data,
model deployment, and prediction request steps. All parts will be completed in the cloud.

#
* **Data description**

The Coastline dataset is accessible via Google Cloud public bucket gs://tamucc_coastline/ and cloud shell:
```
$ gsutil ls gs://tamucc_coastline/
gs://tamucc_coastline/GooglePermissionForImages_20170119.pdf
gs://tamucc_coastline/dict.txt
gs://tamucc_coastline/dict_explanation.csv
gs://tamucc_coastline/esi_images.zip
gs://tamucc_coastline/labeled_images.csv
gs://tamucc_coastline/labels.csv
gs://tamucc_coastline/esi_images/
```
to build a customized image classification model via transfer learning and the existent [Inception-v3 model](https://www.tensorflow.org/tutorials/image_recognition) 
in order to correctly label different types of coastlines using Cloud Machine Learning Engine.  
To adapt data to existing code, file gs://tamucc_coastline/labeled_images.csv must be split into evaluation (10%) and training (90%) sets, then stored to a user created cloud bucket. All references to evaluation and training sets are modified, as well as references to dict.txt (label ground truth file for categorical classification) and example image to be tested. In addition, all references to *flower* are changed to *coastline* which is partly cosmetic and partly to avoid model naming clashes that could happen if flowers example is run in the same context.

* **Disclaimer**

This dataset is provided by a third party. Google provides no representation,
warranty, or other guarantees about the validity or any other aspects of this dataset.

* **Setup and test your GCP environment**

The best way to setup your GCP project is to use this section in this
[tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#set-up-your-gcp-project).

* **Environment setup:**

Virtual environments are strongly suggested, but not required. Installing this
sample's dependencies in a new virtual environment allows you to run the sample
locally without changing global python packages on your system.

There are two options for the virtual environments:

*   Install [Virtualenv](https://virtualenv.pypa.io/en/stable/) 
    *   Create virtual environment `virtualenv myvirtualenv`
    *   Activate env `source myvirtualenv/bin/activate`
*   Install [Miniconda](https://conda.io/miniconda.html)
    *   Create conda environment `conda create --name myvirtualenv python=2.7`
    *   Activate env `source activate myvirtualenv`

* **Install dependencies**

Install the python dependencies. `pip install --upgrade -r requirements.txt`

**Note:** Currently Apache Beam is only supported with [Python 2.7](https://beam.apache.org/get-started/quickstart-py/). 

#

* **How to satisfy Cloud ML Engine project structure requirements**

Follow [this](https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer#project-structure) guide to structure your training application.

# Data processing

You will run sample code in order to preprocess data with Cloud Dataflow and then use that transformed data to train a model with Cloud ML Engine. You will then deploy the trained model to Cloud ML Engine and test the model by sending a prediction request to it.

In this sample dataset you only have a small set of images (~10,533). Without more data it isn’t possible to use machine learning techniques to adequately train an accurate classification model from scratch. Instead, you’ll use an approach called transfer learning. In transfer learning you use a pre-trained model to extract image features that you will use to train a new classifier. In this tutorial in particular you’ll use a pre-trained model called Inception.

```
export PROJECT=$(gcloud config list project --format "value(core.project)")
export JOB_ID="coastlines_${USER}_$(date +%Y%m%d_%H%M%S)"
export BUCKET="gs://${PROJECT}-ml"
export GCS_PATH="${BUCKET}/${USER}/${JOB_ID}"
export DICT_FILE=gs://tamucc_coastline/dict.txt

export MODEL_NAME=coastlines
export VERSION_NAME=v1
```

For the coastline example we need to make a local copy of labeled_images.csv, clean up the extension case (upper or lower case, to match the actual image list on disk, create evaluation and training sets and copy to bucket. 

```
# make local copy
gsutil cp gs://tamucc_coastline/labeled_images.csv .
# get image list
gsutil ls gs://tamucc_coastline/esi_images/ > image_list.txt
# fix extensions
python fix_extension_case.py
# rename
mv labeled_images_fixed.csv labeled_images.csv
# split into evaluation and training sets
python eval_train.py
# copy evaluation and training sets to bucket
gsutil cp *_set.csv ${BUCKET}
# cleanup
rm *.csv
```
* **Use DataFlow to preprocess dataset**

Takes about 30 mins to preprocess everything.  We serialize the two
preprocess.py synchronous calls just for shell scripting ease; you could use
`--runner DataflowRunner` to run them asynchronously.  Typically,
the total worker time is higher when running on Cloud instead of your local
machine due to increased network traffic and the use of more cost efficient
CPU's.  Progress can be monitored on the [Dataflow Console](https://console.cloud.google.com/dataflow)

Pre-process training

```
python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://${PROJECT}-ml/train_set.csv" \
  --output_path "${GCS_PATH}/preproc/train" \
  --cloud
```  
  
Pre-process evaluation

```
python trainer/preprocess.py \
  --input_dict "$DICT_FILE" \
  --input_path "gs://${PROJECT}-ml/eval_set.csv" \
  --output_path "${GCS_PATH}/preproc/eval" \
  --cloud
```

At this stage outputs would have been generated and will be visible in the [Storage Console] (https://console.cloud.google.com/storage/browser).

# Training

* **Google Cloud ML Engine**


* **Run in Google Cloud ML Engine**

Training on CloudML is quick after preprocessing.  If you ran the above
commands asynchronously, make sure they have completed before calling this one.


* **Run in Google Cloud ML Engine:**

```
gcloud ml-engine jobs submit training "$JOB_ID" \
  --stream-logs \
  --module-name trainer.task \
  --package-path trainer \
  --staging-bucket "$BUCKET" \
  --region us-central1 \
  --runtime-version=1.10 \
  -- \
  --output_path "${GCS_PATH}/training" \
  --eval_data_paths "${GCS_PATH}/preproc/eval*" \
  --train_data_paths "${GCS_PATH}/preproc/train*"
```

* **Monitor with TensorBoard:**

```
tensorboard --logdir=${GCS_PATH}/training
```


# Prediction

Remove the model and its version, make sure no error is reported if model does not exist.
```
gcloud ml-engine versions delete $VERSION_NAME --model=$MODEL_NAME -q --verbosity none
gcloud ml-engine models delete $MODEL_NAME -q --verbosity none
```

Once your training job has finished, you can use the exported model to create a prediction server. To do this you first create a model:

```
gcloud ml-engine models create "$MODEL_NAME" \
  --regions us-central1
```


Each unique Tensorflow graph--with all the information it needs to execute--
corresponds to a "version".  Creating a version actually deploys our
Tensorflow graph to a Cloud instance, and gets is ready to serve (predict).

```
gcloud ml-engine versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_PATH}/training/model" \
  --runtime-version=1.10
```

Models do not need a default version, but its a great way move your production
service from one version to another with a single gcloud command.

```
gcloud ml-engine versions set-default "$VERSION_NAME" --model "$MODEL_NAME"
```

* **Run Online Predictions**

You can now send prediction requests to the API. To test this out you can use the `gcloud ml-engine predict` tool:

Download a daisy so we can test online predictions.
```
# TODO ADD DATALAB METHOD (ARRAY INTERATION)
# See https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/quests/scientific/coastline.ipynb
# Section - Deploy and predict model
gsutil cp \
  gs://tamucc_coastline/esi_images/IMG_0001_SecBC_Spr12.jpg \
  IMG_0001_SecBC_Spr12.jpg
```

Since the image is passed via JSON, we have to encode the JPEG string first.

```
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' IMG_0001_SecBC_Spr12.jpg &> request.json
```

Test online prediction

```
gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json
```

You should see a response with the predicted labels of the examples!


## References

[Flowers tutorial](https://cloud.google.com/ml-engine/docs/tensorflow/flowers-tutorial)
