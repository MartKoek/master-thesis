# Gender Fairness in Depression Severity Prediction using Multimodal Models
MSc thesis project by Mart Koek

### Abstract 
Mental health illnesses cause significant suffering for individuals, their families, and society. Early, accurate, and responsible detection of mental health problems is crucial for effective intervention. This research aims to develop responsible methods to assist (not replace) medical experts in depression detection. The performance and gender fairness of uni- and bimodal audio-text models to predict depression severity are explored using the DAIC-WOZ dataset. Three key research questions are addressed: the comparative performance of uni- and bimodal models, the gender fairness of these models, and the effect of bias mitigation methods. The findings indicate that the unimodal text model outperforms state-of-the-art uni- and bimodal audio-text models. The best bimodal models could not improve the performance of our unimodal text model but outperform state-of-the-art bimodal audio-text models. Gender biases were found in all unimodal and bimodal models, with a general trend per modality: text models showed a bias favoring males and audio models favoring females. The bias mitigation methods showed mixed results, sometimes improving fairness but at the cost of overall performance.

### Pipeline of proposed approach
![pipeline_research](https://github.com/MartKoek/master-thesis/assets/59614066/a5890e80-6e69-4def-badb-2298b0356ea2)

### Architecture of models
Three unimodal audio models, one unimodal text model, and one bimodal audio-text model can be constructed with the following architecture:
![unimodal_pipeline](https://github.com/MartKoek/master-thesis/assets/59614066/b2138f82-180d-4367-906c-9cf77fc0dc48)

One bimodal audio-text model can be constructed with the following architecture (feature fusion):
![ff](https://github.com/MartKoek/master-thesis/assets/59614066/9100b91a-89f9-42f6-aea2-7d4a3acaf56b)

Three bimodal audio-text models can be constructed with the following architecture (decision fusion):
![df](https://github.com/MartKoek/master-thesis/assets/59614066/d95ecef2-6a45-4b7e-b38a-16baa2ddb010)

### Organisation of this repository
* The folder preprocessing contains files to prepare the data for the experimentation;
* The folder 7fold_cv contains files to optimize models on the (training + development set) of the DAIC-WOZ;
* The folder evaluation contains files to test the models on the test set of the DAIC-WOZ;
