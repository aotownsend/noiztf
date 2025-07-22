# noiztf
Creates augmented copies of ZTF light curves with higher redshifts and additional noise.

The script ```create_train_test.py``` allows you to create your own augmented training and test sets, from a sample of ZTF light curves in BTS (Bright Transient Survey; Fremling+2020, Perley+2020).

The script ```create_test_alt.py``` allows you to create your own augmented test sets, from an alternative sample of ZTF light curves that are not included in BTS.

Contact Alice Townsend (alice.townsend@physik.hu-berlin.de) for how to access the ZTF light curves, and the corresponding BTS and SLSN data tables required to replicate the work of Townsend+2025.


# Pre-trained ParSNIP models for ZTF

The folder ```pretrained_models``` contains the ParSNIP (Boone2021) models and classifiers used in Townsend+2025.

The files ending in ```.pt``` are the ```parsnip_model``` files and the files ending in ```.pkl``` are the ```parsnip_classifier``` files.

Assuming you have a working installation of ParSNIP and your ```dataset``` is loaded in with ```lcdata```, you can classify data like so:

```
model = parsnip.load_model(parsnip_model, threads=1)
classifier = parsnip.Classifier.load(parsnip_classifier)
predictions = model.predict_dataset(dataset)
classifications = classifier.classify(predictions)
```
