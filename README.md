# CCM-ART-MNIST

Alex, Rob and Taurean's final project for Computational Cognitive Models Course

## Project Details
In our project we are comparing the network representations of 4 different CNN architectures in order to compare how each layer and final representation of each digit determines the similarity of any two same-digit pairs. We will do this by comparing the layer and fully connected layer representations' similarity scores compare to human similarity scores.
 - 1 layer with 2 fully connected layers
 - 2 concatenated layers with 2 fully connected layers
 - 3 concatenated layers with 2 fully connected layers
 - 4 concatenated layers with 2 fully connected layers

 You can find the notebook for performing the hyperparameter tuning [here](./pytorch_hyperparameter_tuning.ipynb). If training the models on your own the data will download the first time through as it is not posted to this repository.
 
 You can find the optimally trained final models [here](./best_models/)


## Information on human similarity ratings
 - Use this [link](https://docs.google.com/spreadsheets/d/1b9reCETO9B3TYaJVgwr6FP2v1wWKT3NJzf6o2cdmc6Q/edit?usp=sharing "Empty Google Sheets Rating Template") to access the Google Sheets rating template.
 - Use the image files [here](./files_for_comparison/) to access the 30 randomly generated digit pairs for each digit. The naming convention is as follows: "digit_comparisonX.png" where X represents a digit from 0 to 9. For example [digit_comparison0.png](./files_for_comparison/digit_comparison0.png) is an image file with all 30 pairs of zeros.
 - Once you have rated the similarity of all 30 pairs on a 1 (not similar at all) to 10 (very similar) scale you can download the Google Sheets document as a .csv file and either post it to [here](./human_ratings/) or email it to one of us.