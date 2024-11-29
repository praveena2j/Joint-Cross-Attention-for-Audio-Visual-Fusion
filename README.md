# Audio–Visual Fusion for Emotion Recognition in the Valence–Arousal Space Using Joint Cross-Attention
Code for our paper "Audio–Visual Fusion for Emotion Recognition in the Valence–Arousal Space Using Joint Cross-Attention" accepted to IEEE T-BIOM 2023. Our paper can be found [here](https://ieeexplore.ieee.org/abstract/document/10005783).

## Citation

If you find this code useful for your research, please cite our paper.

```
@INPROCEEDINGS{10095234,
  author={Praveen, R Gnana and Cardinal, Patrick and Granger, Eric},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={Audio–Visual Fusion for Emotion Recognition in the Valence–Arousal Space Using Joint Cross-Attention}, 
  year={2023},
}
```

This code uses the Affwild2 dataset to validate the proposed approach for Dimensional Emotion Recognition. There are three major blocks in this repository to reproduce the results of our paper. This code uses Mixed Precision Training (torch.cuda.amp). The dependencies and packages required to reproduce the environment of this repository can be found in the `environment.yml` file. 

### Creating the environment
Create an environment using the `environment.yml` file

`conda env create -f environment.yml`

### Models
The pre-trained models of audio and visual backbones are obtained [here](https://github.com/kuhnkeF/ABAW2020TNT)

The fusion models trained using our fusion approach can be found [here](https://drive.google.com/file/d/1Tq1WF6n9uKtkon-vaQ9fjjEeHgNxaS9l/view?usp=drive_link)

```
jointcam_model.pt:  Fusion model trained using our approach on the Affwild2 dataset
```

# Table of contents <a name="Table_of_Content"></a>

+ [Preprocessing](#DP) 
    + [Step One: Download the dataset](#PD)
    + [Step Two: Preprocess the visual modality](#PV) 
    + [Step Three: Preprocess the audio modality](#PA)
    + [Step Four: Preprocess the annotations](#PL)
+ [Training](#Training) 
    + [Training the fusion model](#TE) 
+ [Inference](#R)
    + [Generating the results](#GR)
 
## Preprocessing <a name="DP"></a>
[Return to Table of Content](#Table_of_Content)

### Step One: Download the dataset <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
Please download the following.
  + The dataset for the valence-arousal track can be downloaded [here](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)

### Step Two: Preprocess the visual modality <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
  + The cropped-aligned images are necessary. They are used to form the visual input. They are already provided by the dataset organizers. Otherwise, you may choose to use [OpenFace toolkit](https://github.com/TadasBaltrusaitis/OpenFace/releases) to extract the cropped-aligned images. However, the per-frame success rate is lower compared to the database-provided version.

### Step Three: Preprocess the audio modality <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
  + The audio files are extracted and segmented to generate the corresponding audio files in alignment with the visual files using [mkvextract](https://mkvtoolnix.download/). To generate these audio files, you can use the file Preprocessing/audio_preprocess.py. 

### Step Four: Preprocess the annotations <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
  + The annotations provided by the dataset organizers are preprocessed to obtain the labels of aligned audio and visual files. To generate these audio files, you can use the file Preprocessing/preprocess_labels.py. 

## Training <a name="DP"></a>
[Return to Table of Content](#Table_of_Content)
  + After obtaining the preprocessed audio and visual files along with annotations, we can train the model using the proposed fusion approach using the main.py script.

## Inference <a name="DP"></a>
[Return to Table of Content](#Table_of_Content)
  + The results of the proposed model can be reproduced using the trained model. In order to obtain the predictions on the test set using our proposed model, we can use the test.py.

### 👍 Acknowledgments
Our code is based on [TSAV](https://github.com/kuhnkeF/ABAW2020TNT) and [Recursive-joint-co-attention](https://github.com/tuffr5/Recursive-joint-co-attention)
