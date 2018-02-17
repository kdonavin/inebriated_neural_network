# Inebriated Neural Network

*Using machine learning to classify "Hot" or "Not" batches of New Belgium beer using chemical metrics and beer-tasters' sensory responses*

**Please note that I am unable to share source code for this project because resulting data pipeline and prediction models are the property of New Belgium Brewery**

## Table Of Contents

* [Motivation](#motivation)
* [The Data](#the-data)
* [Analysis](#analysis)
* [Conclusion](#conclusion)
* [Acknowledgments](#acknowledgments)

## Motivation

<img src="images/nb_break_line.png" width=100% height=100%/>

Can we use chemical measurements of New Belgium beer to predict the commentary of trained beer tasters'. Chemical metrics of alcohol (ABV), calories, gravity (Ea), pH and other standard measurements are used to predict whether a batch of beer will be rated "Hot" or "Not" by trained New Belgium beer tasters. "Hot" beer meets expectations in terms of key characteristics, such as clarity, aromas, taste profile, and so on. "Not" beer fails to meet these expectations. Both "Hot" and "Not" are stand-ins for proprietary terminology. 

A web application is developed for use by New Belgium's team of sensory scientists monitoring chemical and biological characteristics of each batch of beer. This tool gives the sensory scientists the capability to predict whether a batch of beer will be given the "Hot" or "Not" stamp of approval by the tasting panel, in order to bottle and ship each batch.

## The Data

<img src="images/nb_break_line.png" width=100% height=100%/>

New Belgium beer-batch data consists of two main datasets, linked on unique brew batch numbers. 

1. Chemical Measurements (batch level): e.g., pH, abv, acetic acid
2. Brand - e.g., Fat Tire, Voodoo Ranger ipa
3. Sensory Data (taster level): "Hot" subjective designations and comments on beer *modalities*: aroma, mouth-feel/body, flavor, aroma and overall/fresh

Data exists at the taster level, which includes the commentary from each taster, as well as at the batch level, which includes chemical measurements of each batch. The taster's "Hot" or "Not" commentary was aggregated from several modality categories to create one "Hot" or "Not" classification per batch. The entire panel is summarized to classify a batch as "Hot" if 90% of votes in the modality categories are "Hot", and "Not" otherwise. 

However, Beer tasters are imperfect and require training in order to be on the panel. As a way to prioritize the most skilled taster's "Hot" or "Not" classifications, *taster quality* metric was used as classification weights prior to aggregation to the batch level. These taster quality measurements were the results of another Galvanize student, Jan Van Zeghbroeck's capstone project [Seeing Taste](https://github.com/janvanzeghbroeck/Seeing-Taste). The quality of each taster is determined by how able they are to detect beer modality differences, and how biased they may be towards "Hot" or "Not" classifications in those modalities. 

## Analysis 

<img src="images/nb_break_line.png" width=100% height=100%/>

<!-- Edit here -->

A **Multi-Layer Perceptron (MLP) Classifier** prediction model from sci-kit learn is applied to this problem. A neural network (in which I taste a hint of vanilla) with 6 Hidden Layers. Model hyper-parameters were selected based on a grid search prioritizing F1-Score (i.e., harmonic mean of precision and recall) for "Hot" or "Not" classification. 

The following is a model performance graphic. The neural network’s performance on "Hot" beer is colored (appropriately) in red and "Not" beer in blue. This model may be useful to New Belgium for it identifies beers that are "Not" in more than 9 out of 10 cases. Model accuracy suffers here because the neural network is essentially over-reactive, guessing "Not" too often. Notice in precision that only about 1 in 5 New Belgium batches labeled "Not" are in fact "Not". however, when the neural network predicts "Hot", it is correct in over 9 of 10 cases.

<img src="images/model_performance.png"/>

## Conclusion

<img src="images/nb_break_line.png" width=100% height=100%/>

This is a proof of concept that chemical measurements of New Belgium brew may be used to predict subjective beer taster commentary.

## Acknowledgments
<img src="images/nb_break_line.png" width=100% height=100%/>

Thank you Matthew Smith, Senior Business Systems Analyst @ New Belgium for data and guidance. Thanks to Jan Van Zechbroeck, [former Galvanize-er](https://github.com/janvanzeghbroeck/Seeing-Taste) and New Belgium consultant for coding help and these attractive section division lines.



