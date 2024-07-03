# Gender Fairness in Depression Severity Prediction using Multimodal Models
MSc thesis project by Mart Koek

## Abstract 
Mental health illnesses cause significant suffering for individuals, their families, and society. Early, accurate, and responsible detection of mental health problems is crucial for effective intervention. This research aims to develop responsible methods to assist (not replace) medical experts in depression detection. The performance and gender fairness of uni- and bimodal audio-text models to predict depression severity are explored using the DAIC-WOZ dataset. Three key research questions are addressed: the comparative performance of uni- and bimodal models, the gender fairness of these models, and the effect of bias mitigation methods. The findings indicate that the unimodal text model outperforms state-of-the-art uni- and bimodal audio-text models. The best bimodal models could not improve the performance of our unimodal text model but outperform state-of-the-art bimodal audio-text models. Gender biases were found in all unimodal and bimodal models, with a general trend per modality: text models showed a bias favoring males and audio models favoring females. The bias mitigation methods showed mixed results, sometimes improving fairness but at the cost of overall performance.

The folder preprocessing contains files to prepare the data for the experimentation.
The folder evaluation contains files with functions to evaluate and finetune the models.
