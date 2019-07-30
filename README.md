# Instance Segmentation

Ce dépôt présente le modèle [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) destiné à la ségmentation d'instances dans des images RGB.
L'implémentation est reprise de [ce dépôt](https://github.com/matterport/Mask_RCNN).

La segmentation d'instance vise à grouper l'ensemble des pixels qui appartiennent à un seul et même objet. Elle est plus complexe que la segmentation
sémantique qui groupe l'ensemble des pixels qui appartiennent à une catégorie d'objets (e.g voitures, personnes,...)

### Requirements

Dans un environnement virtuel créé à l'aide d'Anaconda, les outils nécessaires peuvent être installés grâce aux commandes suivantes: 
```shell 
pip install -r requirements.txt
python setup.py install
```

### Training

Plusieurs exemples d'entraînement sont disponibles dans [le dépôt de base](https://github.com/matterport/Mask_RCNN). 
Ce dépôt propose un nouvel exemple de segmentation sur le dataset [Cityscapes](https://www.cityscapes-dataset.com/). Les archives "gtFine_trainvaltest.zip" et "leftImg8bit_trainvaltest.zip"
sont nécessaires pour l'entraînement.

Dans le cas de Cityscapes, le modèle apprend à segmenter 9 types d'objets: 
* Background
* Person
* Rider
* Car
* Truck
* Bus
* Train
* Motorcycle
* Bike

La commande suivante permet de lancer une procédure de fine-tuning d'un modèle pré-entraîné sur coco:
```shell
python cityscapes.py train --dataset /media/ambroise/cvdatasets/cityscapes/ --model coco --log /path/to/log
```
Les paramètres du modèle pré-entraîné peuvent être téléchargés à [cette adresse](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5).
Le fichier doit être placé à la racine du projet (i.e dans le dossier "InstanceSegmentation"). 


### Prediction 

Le script [prediction_example.py]() permet de tester le modèle sur une image:
```shell
python prediction_example.py --weights path/to/weights/file --image_path path/to/test/image/file
```
Si les dimensions de l'image de test ne correspondent pas à celles du dataset Cityscapes, alors, le script modifie la taille de l'image et la sauve sous le nom "testimage"_resized.png.
La sortie est une image 16-bit avec les différents objets identifiés par le modèle.
Les valeurs attribuées aux pixels des instances sont de la forme "xxyyy" dans laquelle "xx" correspond à l'identifiant de la classe (selon la logique définie [ici](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py))
et "yyy" correspond au numéro de l'instance. Par exemple, pour une image contenant deux voitures, les valeurs possibles des pixels dans le masque seront: "26000" et "26001". L'ordre des voitures est choisi arbitrairement.
