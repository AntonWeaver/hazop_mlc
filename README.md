# Application of Multi-label classification to HAZOP texts

Main idea - using Ml-methods to develop a model to make some predictions based on a text description of a hazardous event in HAZOP worksheets.

HAZOP (Hazard and Operability Study) is a common method for identifying hazards in the operation of existing and in the design of new facilities. The HAZOP procedure is a sequential and routine analysis using the brainstorming method. So, during HAZOP, a large amount of information is recorded in text form, which can later be used to train various models.

The goal of this project is to use machine learning methods (deep learning in particular) to develop a model to predict the expected level of severity of consequences based on a text description of a hazardous event in HAZOP worksheets.

Multi-label* text classification was implemented for 5 categories. 
*(due to the specifics of the task, as well as due to the potential belonging of the text to different categories)

The original dataset includes a mix of data from various HAZOP procedures.

for english-language data BERT was used  (https://huggingface.co/bert-base-cased)
for russian-language data ruBERT was used  (https://huggingface.co/sberbank-ai/ruBert-base)

Main steps:
1. Data collection (previously done)
2. EDA
3. Data preprocessing
4. Text tokenization
5. NN for multi-label classification
6. Model training
7. Validation & tests

Main results:
- 85% Average accuracy (i.e., the proportion of correctly labeled objects) for both for both English and Russian-language data
- 65% Average accuracy on a test object (on new data)

This .ipynb file contains only the code for English-language data.