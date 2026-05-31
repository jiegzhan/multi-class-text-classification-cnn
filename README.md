### Project: Classify Kaggle Consumer Finance Complaints

### Highlights:

 - This is a **multi-class text classification (sentence classification)** problem.
 - The purpose of this project is to **classify Kaggle Consumer Finance Complaints into 11 classes**. 
 - The model was built with **Convolutional Neural Network (CNN)** and **Word Embeddings** on **TensorFlow 2 / Keras**.

### Data: [Kaggle Consumer Finance Complaints](https://www.kaggle.com/cfpb/us-consumer-finance-complaints)

 - Input: **consumer_complaint_narrative**

    - Example: "someone in north Carolina has stolen my identity information and has purchased items including XXXX cell phones thru XXXX on XXXX/XXXX/2015. A police report was filed as soon as I found out about it on XXXX/XXXX/2015. A investigation from XXXX is under way thru there fraud department and our local police department.\n"
    
 - Output: **product**

     - Example: Credit reporting

### Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train:

 - Command: `python train.py <data_file> <params_file>`
 - Example: `python train.py ./data/consumer_complaints.csv.zip ./parameters.json`
 
 A directory (`trained_model_<timestamp>/`) will be created during training:
 - `best_model.keras` — model with best validation accuracy
 - `final_model.keras` — model at end of training
 - `vectorizer.keras` — text vectorization layer (vocabulary)
 - `train_config.json` — training metadata and label mapping

### Predict:

 Provide the model directory (created when running `train.py`) and new data to `predict.py`.
 - Command: `python predict.py <model_directory> <test_data.json>`
 - Example: `python predict.py ./trained_model_1479757124/ ./data/small_samples.json`

 Predictions are saved to `./data/predictions_output.json`.

### Reference:
 - [Implement a CNN for text classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
