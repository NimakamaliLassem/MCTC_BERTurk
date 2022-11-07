> **T.C. Adalet Bakanlığı Bilgi İşlem Genel Müdürlüğü**
>
> OCTOBER 24, 2022
>
> Multi-Class and Multi-Label Text Classification\
> **Project Plan**
>
> 13 JUNE TO 26 DECEMBER 2020
>
> PREPARED BY:\
> BEYZANAZ CAMCIOĞLU\
> NIMA KAMALI LASSEM
>
> Table of Contents

+---------------------------------------------+----+
| > Research (June 13 - October 14)           | 2  |
+=============================================+====+
| > School (September 15 - January 13)        | 3  |
+---------------------------------------------+----+
| > Project (October 11 - December 23)        | 3  |
+---------------------------------------------+----+
| > Research and School Gantt Charts          | 5  |
+---------------------------------------------+----+
| > Project Gantt Chart                       | 6  |
+---------------------------------------------+----+
| > **Multi-Class Text Classiﬁcation Steps:** | 7  |
+---------------------------------------------+----+
| > 1\. Install Tensorﬂow Library             | 7  |
+---------------------------------------------+----+
| > 2\. Install Transformers Library          | 7  |
+---------------------------------------------+----+
| > 3\. Import Necessary Libraries            | 7  |
+---------------------------------------------+----+
| > 4\. Create a suitable Dataset             | 7  |
+---------------------------------------------+----+
| > 5\. Read the Dataset into a Dataframe     | 8  |
+---------------------------------------------+----+
| > 6\. Prepare Data                          | 8  |
+---------------------------------------------+----+
| > 7\. Prepare Model                         | 12 |
+---------------------------------------------+----+
| > 8\. Using the model                       | 14 |
+---------------------------------------------+----+
| > **Multi-Label Text Classiﬁcation**        | 16 |
+---------------------------------------------+----+
| > References                                | 17 |
+---------------------------------------------+----+

2

> Internship Project Plan
>
> Beyzanaz Camcıoğlu and Nima Kamali Lassem internship from June 13 to
> December 26, 2022.

![](vertopal_f4c0706f74e74041915764d82d41ebe1/media/image1.png){width="7.083333333333333in"
height="1.5in"}

> The following project plan consists of three main sections:

  1\.   1\. Research: The research phase of the project we have been performing since June 13, 2022.
  ----- ----------------------------------------------------------------------------------------------
  2\.   2\. School: The school (CTIS) department-related tasks like follow-up meetings started on

> September 15, 2022.

  ----- ----------------------------------------------------------------------------------------------------
  3\.   3\. Project: The steps we predict we will take toward completing the project by December 23, 2022.
  ----- ----------------------------------------------------------------------------------------------------

> Research (June 13 - October 14)

![](vertopal_f4c0706f74e74041915764d82d41ebe1/media/image2.png){width="7.083333333333333in"
height="1.551388888888889in"}

  --- -----------------------------------------------------------------------------------------------
  ●   ● **Researching NLP/Models:** Researching NLP concepts and their models to determine the most
  --- -----------------------------------------------------------------------------------------------

> suitable for domain-speciﬁc problems such as medical and justice
> domains. The studies includeda search for Train and Test datasets.

  --- ---------------------------------------------------------------------------------------------------------
  ●   ● **Experiment Data Collection**: to fulﬁll the need for related datasets for the upcoming experiments,
  --- ---------------------------------------------------------------------------------------------------------

> we collected data from various sources, including Kaggle, Twitter, and
> legal documents. We have
>
> covered more than 250,000 court decisions (mahkeme kararları) and
> approximately 5 Gigabytes
>
> of public formal (Yargıtay and Danıştay) and informal (Turkish Spam
> messages) documents.

3

  --- ------------------------------------------------------------------------------------------------
  ●   ● **Pre-project Experimenting:** to prove our ﬁndings from our research on NLP, we carried out
  --- ------------------------------------------------------------------------------------------------

> experiments on different language models like BERT and fastText. After
> successfully comparingthe said models and displaying BERT's lead on
> the experiment results, we have investigated itsfunctions and possible
> outcomes. Up to now, BERTurk-Base-128k has shown promising resultsfor
> Binary and Multi-class Text Classiﬁcations with accuracy scores of
> over 95 and 75 percent,respectively.

  ------- -----------------------------------------------------------------------------------------------
  **●**   **●** **Hardware Requirements:** our research showed a minimum GPU RAM size of 16 GB. We were
  ------- -----------------------------------------------------------------------------------------------

> given remote access to a desktop computer equipped with an RTX A6000
> 48 GB GPU. We will
>
> request assistance in case of a need for more computing power for the
> project.
>
> School (September 15 - January 13)
>
> ![](vertopal_f4c0706f74e74041915764d82d41ebe1/media/image3.png){width="7.1125in"
> height="2.4902777777777776in"}

  ●   ● **Follow-Up Meetings:** consist of ﬁve assignments regarding work experience.
  --- ------------------------------------------------------------------------------------------------------
  ●   ● **Interim Meeting:** Presentation of acquired skills and experience to the internship coordinator.
  ●   ● **Company Evaluation and Report Evaluation:** Evaluations forms. 30 Hour workload expected.
  ●   ● **Jury:** Presentation of acquired skills and experience to the jury members after the

> internship---January 7th to 13th.
>
> Project (October 11 - December 23)

+----------------------------------+----------------------------------+
| > **Requirements gathering**     | ![](vertopal_f4c0706f74e740      |
|                                  | 41915764d82d41ebe1/media/image4. |
|                                  | png){width="5.763888888888889in" |
|                                  | height="2.2708333333333335in"}   |
+==================================+==================================+
| > **Data Collection and          |                                  |
| > Processing**                   |                                  |
+----------------------------------+----------------------------------+
| > **Code and Test Weeks**        |                                  |
+----------------------------------+----------------------------------+
| > **Deployment / Final Report**  |                                  |
+----------------------------------+----------------------------------+

4

  --- -------------------------------------------------------------------------------------------------
  ●   ● **Requirements Gathering:** determination and collection of the functional and non-functional
  --- -------------------------------------------------------------------------------------------------

> requirements for the project. Understanding the project aims and goals
> and choosing resourcesand tools to be used for the projects. This task
> will be accomplished alongside or immediatelyfollowed by Data
> Collection and Processing.

  ------- ------------------------------------------------------------------------------------------------------------
  **●**   **●** **Data Collection and Processing:** similar to the data collection in the experiment phase. As there
  ------- ------------------------------------------------------------------------------------------------------------

> are a limited number of resources available in the justice domain, it
> is expected that we will needa long time to create or process the
> previously obtained data to be usable as *"train and testdatasets"*
> for the project A labeling should be done within the scope of the
> multi-class textclassiﬁcation project. We have considered that a
> labeling application should be made for thecorrect labeling of the
> data, and this application should be designed for labeling processes
> of lawdocuments. We found that the most appropriate structure for
> training is the design of documentsby breaking them down into
> meaningful sequences and labeling the phrases in accordance withthe
> sentences they belong. Since BERT works with a limited amount of words
> in the sequence, weneed to process our data accordingly. The data will
> be stored in TSV (tab-separated values)format in two main columns:
> sequences and categories (classes). There can be more column inthe
> dataset, however, they will not be used in the ﬁnal program.

  ------- -----------------------------------------------------------------------------------------------------------------
  **●**   **●** **Code and Test weeks:** we will deliver the project in multiple iterations, facilitating weekly tests to
  ------- -----------------------------------------------------------------------------------------------------------------

> ensure admissible project results. At the end of each week (Fridays),
> we will report the week'ssuccess rate evaluation and issues to be
> dealt with in the sprint. We will repeat this task until itproduces
> satisfying results or enters the deployment phase.

  ------- ----------------------------------------------------------------------------------------------------------------
  **●**   **●** **Deployment / Final Report:** At the end of the internship, we will take action based on the project\'s
  ------- ----------------------------------------------------------------------------------------------------------------

> ﬁnal results. This phase includes saving and deploying the most
> successful model.
>
> **The details on this document are subject to change. Adjustments
> are**
>
> **to be performed in the upcoming days.**

5

> **Research (June 13 - October 14) - Gantt**

![](vertopal_f4c0706f74e74041915764d82d41ebe1/media/image2.png){width="10.511111111111111in"
height="2.3222222222222224in"}

> **School (September 15 - January 13) - Gantt**

![](vertopal_f4c0706f74e74041915764d82d41ebe1/media/image3.png){width="10.511111111111111in"
height="3.6666666666666665in"}

6

> **Project (October 11 - December 23) - Gantt**

![](vertopal_f4c0706f74e74041915764d82d41ebe1/media/image5.png){width="11.280555555555555in"
height="5.5527766841644794in"}

7

> Multi-Class Text Classiﬁcation Steps:

+--------+---------------------------------+
| **1.** | > **Install Tensorﬂow Library** |
+--------+---------------------------------+

> To make sure the TensorFlow library will have access to the GPU, we
> need to sync the Python,cudnn, and CUDA versions. Up to now, October
> 24th 2022, the latest matching versions of eachrequirement.

+----------+----------+----------+----------+----------+----------+
| > **V    | **Python | > **Co   | **Build  | *        | >        |
| ersion** | v        | mpiler** | tools**  | *cuDNN** | **CUDA** |
|          | ersion** |          |          |          |          |
+==========+==========+==========+==========+==========+==========+
| >        | >        | GCC      | Bazel    | 8.1      | > 11.2   |
| tensorﬂo | 3.7-3.10 | 9.3.1    | 5.1.1    |          |          |
| w-2.10.0 |          |          |          |          |          |
+----------+----------+----------+----------+----------+----------+

> **2. Install Transformers Library**

+---------------------------------+
| > **!pip install transformers** |
+---------------------------------+

> **3. Import Necessary Libraries**\
> Tensorﬂow or PyTorch can be used in the project. As a result of the
> experiments, the functionsused by Tensorﬂow to ﬁt the data into the
> model were found to be more eﬃcient. The Pandaslibrary will be used to
> read the data from the ﬁle and store it in a data structure. Numpy
> library tocreate and manage arrays quicker.

+----------------------------------+
| > **import pandas as pd**        |
+==================================+
| > **import numpy as np**         |
+----------------------------------+
| > **from tqdm.auto import tqdm** |
+----------------------------------+
| > **import tensorflow as tf**    |
+----------------------------------+

> **4. Create a suitable Dataset**\
> BERT's default labeling format is to have items (sentence, category)
> in the ﬁrst row and thecolumn order as "label" and "sentence"
> respectively. As our program will automatically assignbinary labels to
> the dataset it is better to form our dataset columns like "sentence,
> category":

8

+-------+-----------------------------------------------+------------+
| > 1   | > sentence                                    | > category |
+=======+===============================================+============+
| > 2   | > A series of escapades demonstrating the     | 1          |
|       | > adage that what is good for the gooseis     |            |
|       | > also good for the gander , some of which    |            |
|       | > occasionally amuses but none ofwhich        |            |
|       | > amounts to much of a story.                 |            |
+-------+-----------------------------------------------+------------+
| > 3   | > A positively thrilling combination of       | 3          |
|       | > ethnography and all the intrigue , betrayal |            |
|       | > ,deceit and murder of a Shakespearean       |            |
|       | > tragedy or a juicy soap opera .             |            |
+-------+-----------------------------------------------+------------+
| > ... | > ...                                         | > ...      |
+-------+-----------------------------------------------+------------+

> **5. Read the Dataset into a Dataframe**
>
> Since the document type is TSV, the separator should be deﬁned as
> '\\t' while using the read_csv
>
> function.

+---------------------------------------------------+
| > **df = pd.read_csv(DATASET_PATH, sep=\'\\t\')** |
+---------------------------------------------------+

> **6. Prepare Data**

+----------------------------------+----------------------------------+
| ●                                | > Drop Unused Columns            |
+==================================+==================================+
| > df.drop(118202,                |                                  |
| > inplace=**True**)              |                                  |
+----------------------------------+----------------------------------+
| > df\[\'category\'\] =           |                                  |
| > df\[\'category\'\].astype(int) |                                  |
+----------------------------------+----------------------------------+
| ●                                | ● Set int values to the          |
|                                  | categories (in case they are not |
|                                  | int64)                           |
+----------------------------------+----------------------------------+
| > \'\'\'                         |                                  |
+----------------------------------+----------------------------------+
| > We have 5 classes/categories   |                                  |
| > in this example                |                                  |
+----------------------------------+----------------------------------+
| > \'\'\'                         |                                  |
+----------------------------------+----------------------------------+
| > d                              |                                  |
| f\[\'category\'\].value_counts() |                                  |
+----------------------------------+----------------------------------+
| > Out\[34\]:                     |                                  |
+----------------------------------+----------------------------------+
| 2.0                              | > 60920 ← NUMBER OF SENTENCES IN |
|                                  | > CATEGORY (CLASS) \#2           |
+----------------------------------+----------------------------------+
| 3.0                              | > 24903                          |
+----------------------------------+----------------------------------+
| 1.0                              | > 20160                          |
+----------------------------------+----------------------------------+
| 4.0                              | > 6884                           |
+----------------------------------+----------------------------------+
| 0.0                              | > 5335                           |
+----------------------------------+----------------------------------+
| > Name: Sentiment, dtype: int64  |                                  |
+----------------------------------+----------------------------------+
| ●                                | ● Download your preferred        |
|                                  | model's Tokenizer                |
+----------------------------------+----------------------------------+
| > tokenizer =                    |                                  |
| > BertT                          |                                  |
| okenizer.from_pretrained(\'dbmdz |                                  |
| /bert-base-turkish-128k-cased\') |                                  |
+----------------------------------+----------------------------------+

9

+----------------------------------+----------------------------------+
| ●                                | ● Create "Input IDs" and         |
|                                  | "Attention Mask" tensors for the |
|                                  | tokenized data                   |
+==================================+==================================+
| > sequence_size = 256            |                                  |
+----------------------------------+----------------------------------+
| > X_input_ids =                  |                                  |
| > np.zeros((len(df),             |                                  |
| > sequence_size))                |                                  |
+----------------------------------+----------------------------------+
| > X_attn_masks =                 |                                  |
| > np.zeros((len(df),             |                                  |
| > sequence_size))                |                                  |
+----------------------------------+----------------------------------+
| ●                                | > Create Tokenizer Module        |
+----------------------------------+----------------------------------+
| > **def                          |                                  |
| > generate_training_data**(df,   |                                  |
| > ids, masks, tokenizer):        |                                  |
+----------------------------------+----------------------------------+
| > **for** i, text **in**         |                                  |
| > tqdm                           |                                  |
| (enumerate(df\[\'sentence\'\])): |                                  |
+----------------------------------+----------------------------------+
| > tokenized_text =               |                                  |
| > tokenizer.encode_plus(         |                                  |
+----------------------------------+----------------------------------+
| > text,                          |                                  |
+----------------------------------+----------------------------------+
| > max_length=sequence_size ,     |                                  |
+----------------------------------+----------------------------------+
| > truncation=**True**,           |                                  |
+----------------------------------+----------------------------------+
| > padding=\'max_length\',        |                                  |
+----------------------------------+----------------------------------+
| > add_special_tokens=**True**,   |                                  |
+----------------------------------+----------------------------------+
| > return_tensors=\'tf\'          |                                  |
+----------------------------------+----------------------------------+
| > )                              |                                  |
+----------------------------------+----------------------------------+
| > ids\[i, :\] =                  |                                  |
| > tokenized_text.input_ids       |                                  |
+----------------------------------+----------------------------------+
| > masks\[i, :\] =                |                                  |
| > tokenized_text.attention_mask  |                                  |
+----------------------------------+----------------------------------+
| > **return** ids, masks          |                                  |
+----------------------------------+----------------------------------+
| ●                                | > Tokenize Dataset               |
+----------------------------------+----------------------------------+
| > X_input_ids, X_attn_masks =    |                                  |
| > generate_training_data(df,     |                                  |
| > X_input_ids, X_attn_masks,     |                                  |
+----------------------------------+----------------------------------+
| > tokenizer)                     |                                  |
+----------------------------------+----------------------------------+
| ●                                | > Create "labels" tensor         |
+----------------------------------+----------------------------------+
| > number_of_classes = 5          |                                  |
+----------------------------------+----------------------------------+
| > labels = np.zeros((len(df),    |                                  |
| > number_of_classes))            |                                  |
+----------------------------------+----------------------------------+
| > labels.shape                   |                                  |
+----------------------------------+----------------------------------+
| > Out\[54\]:                     |                                  |
+----------------------------------+----------------------------------+
| > (118202, 5)                    |                                  |
+----------------------------------+----------------------------------+
| ●                                | > One-Hot encoded target tensor  |
+----------------------------------+----------------------------------+
| > In \[61\]:                     |                                  |
+----------------------------------+----------------------------------+
| >                                |                                  |
|  labels\[np**.**arange(len(df)), |                                  |
| >                                |                                  |
| df\[\'Sentiment\'\]**.**values\] |                                  |
| > **=** 1                        |                                  |
+----------------------------------+----------------------------------+

> "One hot encoding is one method of converting data to prepare it for
> an algorithm and get a betterprediction. With one-hot, we convert each
> categorical value into a new categorical column and assign abinary
> value of 1 or 0 to those columns. Each integer value is represented as
> a binary vector. All thevalues are zero, and the index is marked with
> a 1."

10

> Take a look at this chart for a better understanding:

![](vertopal_f4c0706f74e74041915764d82d41ebe1/media/image6.png){width="7.083333333333333in"
height="1.8430544619422573in"}

> Or in our case:

+----------------------------------+----------------------------------+
| > labels                         |                                  |
+==================================+==================================+
| > Out\[61\]:                     |                                  |
+----------------------------------+----------------------------------+
| > array(\[\[0., 1., 0., 0.,      |                                  |
| > 0.\],                          |                                  |
+----------------------------------+----------------------------------+
| > \[0., 0., 1., 0., 0.\],        |                                  |
+----------------------------------+----------------------------------+
| > \[0., 0., 1., 0., 0.\],        |                                  |
+----------------------------------+----------------------------------+
| > \...,                          |                                  |
+----------------------------------+----------------------------------+
| > \[0., 1., 0., 0., 0.\],        |                                  |
+----------------------------------+----------------------------------+
| > \[0., 0., 0., 0., 1.\],        |                                  |
+----------------------------------+----------------------------------+
| > \[0., 0., 0., 0., 1.\]\])      |                                  |
+----------------------------------+----------------------------------+
| ●                                | Create a data pipeline using     |
|                                  | tensorﬂow dataset utility,       |
|                                  | creates batches of data for easy |
|                                  | loading\...                      |
+----------------------------------+----------------------------------+
| > dataset =                      |                                  |
| > tf.data.Dataset.               |                                  |
| from_tensor_slices((X_input_ids, |                                  |
| > X_attn_masks, labels))         |                                  |
+----------------------------------+----------------------------------+
| > dataset.take(1) *\# one sample |                                  |
| > data*                          |                                  |
+----------------------------------+----------------------------------+
| > Out\[66\]:                     |                                  |
+----------------------------------+----------------------------------+
| > \<TakeDataset shapes: ((256,), |                                  |
| > (256,), (5,)), types:          |                                  |
| > (tf.float64, tf.float64,       |                                  |
+----------------------------------+----------------------------------+
| > tf.float64)\>                  |                                  |
+----------------------------------+----------------------------------+
| ●                                | > Create a Map Module            |
+----------------------------------+----------------------------------+
| > **def                          |                                  |
| > MCT                            |                                  |
| CDatasetMapFunction**(input_ids, |                                  |
| > attn_masks, labels):           |                                  |
+----------------------------------+----------------------------------+
| > **return** {                   |                                  |
+----------------------------------+----------------------------------+
| > \'input_ids\': input_ids,      |                                  |
+----------------------------------+----------------------------------+
| > \'attention_mask\': attn_masks |                                  |
+----------------------------------+----------------------------------+
| > }, labels                      |                                  |
+----------------------------------+----------------------------------+

11

+----------------------------------+----------------------------------+
| ●                                | > Perform Mapping                |
+==================================+==================================+
| > *dataset =                     |                                  |
| > data                           |                                  |
| set.map(MCTCDatasetMapFunction)* |                                  |
+----------------------------------+----------------------------------+
| > *dataset.take(1)*              |                                  |
+----------------------------------+----------------------------------+
| > Out\[69\]:                     |                                  |
+----------------------------------+----------------------------------+
| > *\<TakeDataset shapes:         |                                  |
| > ({input_ids: (256,),           |                                  |
| > attention_mask: (256,)},       |                                  |
| > (5,)), types:*                 |                                  |
+----------------------------------+----------------------------------+
| > *({input_ids: tf.float64,      |                                  |
| > attention_mask: tf.float64},   |                                  |
| > tf.float64)\>*                 |                                  |
+----------------------------------+----------------------------------+
| ●                                | ● Set batch size, drop any left  |
|                                  | out tensor, shuﬄe the dataset to |
|                                  | improve accuracy.                |
+----------------------------------+----------------------------------+

> Shuﬄing is done so that BERT does not try to ﬁnd a connection between
> the sequences that werecreated from the same sentence. For example, "I
> will not sleep tonight" can be broken down to A("sleep tonight"), and
> B ("I will not"). There is no connection between A and B but if they
> are putnext to each other BERT might think there has to be a
> connection.

+----------------------------------+----------------------------------+
| > dataset =                      |                                  |
| >                                |                                  |
| dataset.shuffle(10000).batch(16, |                                  |
| > drop_remainder=**True**)       |                                  |
+==================================+==================================+
| ●                                | Divide your dataset to Train and |
|                                  | Validation sets (Recommend the   |
|                                  | Train set to be %80 of the       |
|                                  | total)                           |
+----------------------------------+----------------------------------+
| +-----------+                    |                                  |
| | > p = 0.8 |                    |                                  |
| +-----------+                    |                                  |
+----------------------------------+----------------------------------+
| train_size =                     |                                  |
| int((len(df)//16)\*p) *\# for    |                                  |
| each 16 batch of data we will    |                                  |
| have len(df)//16*                |                                  |
+----------------------------------+----------------------------------+
| > *samples, take 80% of that for |                                  |
| > train.*                        |                                  |
+----------------------------------+----------------------------------+
| > In \[75\]:                     |                                  |
+----------------------------------+----------------------------------+
| > train_size                     |                                  |
+----------------------------------+----------------------------------+
| > Out\[75\]:                     |                                  |
+----------------------------------+----------------------------------+
| > 5909                           |                                  |
+----------------------------------+----------------------------------+
| > In \[76\]:                     |                                  |
+----------------------------------+----------------------------------+
| > train_dataset =                |                                  |
| > dataset.take(train_size)       |                                  |
+----------------------------------+----------------------------------+
| > val_dataset =                  |                                  |
| > dataset.skip(train_size)       |                                  |
+----------------------------------+----------------------------------+

12

> **7. Prepare Model**

+----------------------+----------------------+----------------------+
| ●                    | > Download your      |                      |
|                      | > preferred Model    |                      |
+======================+======================+======================+
| > ***from**          |                      |                      |
| > transformers       |                      |                      |
| > **import**         |                      |                      |
| > TFBertModel*       |                      |                      |
+----------------------+----------------------+----------------------+
| > *In \[80\]:*       |                      |                      |
+----------------------+----------------------+----------------------+
| > *model =           |                      |                      |
| > TFBertMode         |                      |                      |
| l.from_pretrained(*\ |                      |                      |
| 'dbmdz/bert-base-tur |                      |                      |
| kish-128k-cased\'*)* |                      |                      |
+----------------------+----------------------+----------------------+
| > *\# BERTurk base   |                      |                      |
| > model 128k with    |                      |                      |
| > pretrained         |                      |                      |
| > weights*           |                      |                      |
+----------------------+----------------------+----------------------+
| ●                    | ● Deﬁne Layers for   |                      |
|                      | input_ids and        |                      |
|                      | attn_masks           |                      |
+----------------------+----------------------+----------------------+
| > input_ids =        |                      |                      |
| > tf.ke              |                      |                      |
| ras.layers.Input(sha |                      |                      |
| pe=(sequence_size,), |                      |                      |
| >                    |                      |                      |
|  name=\'input_ids\', |                      |                      |
+----------------------+----------------------+----------------------+
| > dtype=\'int32\')   |                      |                      |
+----------------------+----------------------+----------------------+
| > attn_masks =       |                      |                      |
| > tf.ke              |                      |                      |
| ras.layers.Input(sha |                      |                      |
| pe=(sequence_size,), |                      |                      |
| > name               |                      |                      |
| =\'attention_mask\', |                      |                      |
+----------------------+----------------------+----------------------+
| > dtype=\'int32\')   |                      |                      |
+----------------------+----------------------+----------------------+
| > bert_embds =       |                      |                      |
| > m                  |                      |                      |
| odel.bert(input_ids, |                      |                      |
| > attention_m        |                      |                      |
| ask=attn_masks)\[1\] |                      |                      |
| > *\# 0 -\>          |                      |                      |
| > activation*        |                      |                      |
+----------------------+----------------------+----------------------+
| > *layer (3D), 1 -\> |                      |                      |
| > pooled output      |                      |                      |
| > layer (2D)*        |                      |                      |
+----------------------+----------------------+----------------------+
| > intermediate_layer |                      |                      |
| > =                  |                      |                      |
| >                    |                      |                      |
|  tf.keras.layers.Den |                      |                      |
| se(sequence_size\*2, |                      |                      |
| >                    |                      |                      |
| activation=\'relu\', |                      |                      |
+----------------------+----------------------+----------------------+
| >                    |                      |                      |
| name=\'intermediate_ |                      |                      |
| layer\')(bert_embds) |                      |                      |
+----------------------+----------------------+----------------------+
| > output_layer =     |                      |                      |
| >                    |                      |                      |
| tf.keras.layers.Dens |                      |                      |
| e(number_of_classes, |                      |                      |
| > act                |                      |                      |
| ivation=\'softmax\', |                      |                      |
+----------------------+----------------------+----------------------+
| > na                 |                      |                      |
| me=\'output_layer\') |                      |                      |
| (intermediate_layer) |                      |                      |
| > *\# softmax -\>    |                      |                      |
| > calcs probs of     |                      |                      |
| > classes*           |                      |                      |
+----------------------+----------------------+----------------------+
| > mctc_model =       |                      |                      |
| > tf.keras.Model     |                      |                      |
| (inputs=\[input_ids, |                      |                      |
| > attn_masks\],      |                      |                      |
| > o                  |                      |                      |
| utputs=output_layer) |                      |                      |
+----------------------+----------------------+----------------------+
| >                    |                      |                      |
| mctc_model.summary() |                      |                      |
+----------------------+----------------------+----------------------+
| > *Out \[88\]:*      |                      |                      |
+----------------------+----------------------+----------------------+
| > Model: \"model\"   |                      |                      |
+----------------------+----------------------+----------------------+
| \_\_\_\_\_\_         |                      |                      |
| \_\_\_\_\_\_\_\_\_\_ |                      |                      |
| \_\_\_\_\_\_\_\_\_\_ |                      |                      |
| \_\_\_\_\_\_\_\_\_\_ |                      |                      |
| \_\_\_\_\_\_\_\_\_\_ |                      |                      |
| \_\_\_\_\_\_\_\_\_\_ |                      |                      |
| \_\_\_\_\_\_\_\_\_\_ |                      |                      |
| \_\_\_\_\_\_\_\_\_\_ |                      |                      |
| \_\_\_\_\_\_\_\_\_\_ |                      |                      |
+----------------------+----------------------+----------------------+
| > Layer (type)       |                      |                      |
| > Output Shape Param |                      |                      |
| > *\# Connected to*  |                      |                      |
+----------------------+----------------------+----------------------+
| ======               |                      |                      |
| ==================== |                      |                      |
| ==================== |                      |                      |
| ==================== |                      |                      |
| ==================== |                      |                      |
+----------------------+----------------------+----------------------+
| > input_ids          | > 0                  | > \[\]               |
| > (InputLayer)       |                      |                      |
| > \[(**None**,       |                      |                      |
| > 256)\]             |                      |                      |
+----------------------+----------------------+----------------------+
| > attention_mask     | > 0                  | > \[\]               |
| > (InputLayer)       |                      |                      |
| > \[(**None**,       |                      |                      |
| > 256)\]             |                      |                      |
+----------------------+----------------------+----------------------+
| > bert               | 108310272            | > \[\'in             |
| > (TFBertMainLayer)  |                      | put_ids\[0\]\[0\]\', |
| >                    |                      |                      |
|  TFBaseModelOutputWi |                      |                      |
+----------------------+----------------------+----------------------+
| thPoolingAndCrossAt  |                      |                      |
+----------------------+----------------------+----------------------+
| > \'attentio         |                      |                      |
| n_mask\[0\]\[0\]\'\] |                      |                      |
+----------------------+----------------------+----------------------+
| tentions(last_hidde  |                      |                      |
+----------------------+----------------------+----------------------+
| n_state=(**None**,   |                      |                      |
| 256,                 |                      |                      |
+----------------------+----------------------+----------------------+
| 768),                |                      |                      |
+----------------------+----------------------+----------------------+

13

+-----------------------------+--------+--------------------------+
| pooler_output=(Non          |        |                          |
+=============================+========+==========================+
| e, 768),                    |        |                          |
+-----------------------------+--------+--------------------------+
| past_key_values=No          |        |                          |
+-----------------------------+--------+--------------------------+
| ne, hidden_states=N         |        |                          |
+-----------------------------+--------+--------------------------+
| one, attentions=Non         |        |                          |
+-----------------------------+--------+--------------------------+
| e, cross_attentions         |        |                          |
+-----------------------------+--------+--------------------------+
| =**None**)                  |        |                          |
+-----------------------------+--------+--------------------------+
| > intermediate_layer        | 393728 | > \[\'bert\[0\]\[1\]\'\] |
| > (Dense) (**None**, 512)   |        |                          |
+-----------------------------+--------+--------------------------+
| > output_layer (Dense)      | > 2565 |                          |
| > (**None**, 5)             |        |                          |
+-----------------------------+--------+--------------------------+
| > \[\'inter                 |        |                          |
| mediate_layer\[0\]\[0\]\'\] |        |                          |
+-----------------------------+--------+--------------------------+
| =====                       |        |                          |
| =========================== |        |                          |
| =========================== |        |                          |
| =========================== |        |                          |
+-----------------------------+--------+--------------------------+
| > Total params: 108,706,565 |        |                          |
+-----------------------------+--------+--------------------------+
| > Trainable params:         |        |                          |
| > 108,706,565               |        |                          |
+-----------------------------+--------+--------------------------+
| > Non-trainable params: 0   |        |                          |
+-----------------------------+--------+--------------------------+
| \_\_\_\_\_                  |        |                          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\ |        |                          |
| _\_\_\_\_\_\_\_\_\_\_\_\_\_ |        |                          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\ |        |                          |
| _\_\_\_\_\_\_\_\_\_\_\_\_\_ |        |                          |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\ |        |                          |
| _\_\_\_\_\_\_\_\_\_\_\_\_\_ |        |                          |
+-----------------------------+--------+--------------------------+

14

+----------------------------------+----------------------------------+
| ●                                | ● Set "Optimization", "Loss",    |
|                                  | and "Accuracy" modules available |
|                                  | on the tensorﬂow library         |
+==================================+==================================+
| > optim =                        |                                  |
| > tf.keras.opt                   |                                  |
| imizers.Adam(learning_rate=1e-5, |                                  |
| > decay=1e-6)                    |                                  |
+----------------------------------+----------------------------------+
| > loss_func =                    |                                  |
| > tf.keras.                      |                                  |
| losses.CategoricalCrossentropy() |                                  |
+----------------------------------+----------------------------------+
| > acc =                          |                                  |
| > tf.keras.metrics.C             |                                  |
| ategoricalAccuracy(\'accuracy\') |                                  |
+----------------------------------+----------------------------------+
| ●                                | > Save the model                 |
+----------------------------------+----------------------------------+
| >                                |                                  |
|  MCTC_model.save(\'MCTC_model\') |                                  |
+----------------------------------+----------------------------------+

> **8. Using the model**

+----------------------------------+----------------------------------+
| ●                                | > Load the Model and its         |
|                                  | > Tokenizer                      |
+==================================+==================================+
| > MCTC_model =                   |                                  |
| > tf.keras.m                     |                                  |
| odels.load_model(\'MCTC_model\') |                                  |
+----------------------------------+----------------------------------+
| > tokenizer =                    |                                  |
| > BertT                          |                                  |
| okenizer.from_pretrained(\'dbmdz |                                  |
| /bert-base-turkish-128k-cased\') |                                  |
+----------------------------------+----------------------------------+
| ●                                | > Create a Data Preparation      |
|                                  | > Module                         |
+----------------------------------+----------------------------------+
| +----------------------------+   |                                  |
| | > **def                    |   |                                  |
| | >                          |   |                                  |
| | prepare_data**(input_text, |   |                                  |
| | > tokenizer):              |   |                                  |
| +----------------------------+   |                                  |
+----------------------------------+----------------------------------+
| > token = tokenizer.encode_plus( |                                  |
+----------------------------------+----------------------------------+
| > input_text,                    |                                  |
+----------------------------------+----------------------------------+
| > max_length=256,                |                                  |
+----------------------------------+----------------------------------+
| > truncation=**True**,           |                                  |
+----------------------------------+----------------------------------+
| > padding=\'max_length\',        |                                  |
+----------------------------------+----------------------------------+
| > add_special_tokens=**True**,   |                                  |
+----------------------------------+----------------------------------+
| > return_tensors=\'tf\'          |                                  |
+----------------------------------+----------------------------------+
| > )                              |                                  |
+----------------------------------+----------------------------------+
| > **return** {                   |                                  |
+----------------------------------+----------------------------------+
| > \'input_ids\':                 |                                  |
| > tf.cast(token.input_ids,       |                                  |
| > tf.float64),                   |                                  |
+----------------------------------+----------------------------------+
| > \'attention_mask\':            |                                  |
| > tf.cast(token.attention_mask,  |                                  |
| > tf.float64)                    |                                  |
+----------------------------------+----------------------------------+
| > }                              |                                  |
+----------------------------------+----------------------------------+

15

+----------------------------------+----------------------------------+
| ●                                | ● Prepare a Predict Module       |
|                                  | (Enter Class names)              |
+==================================+==================================+
| > **def make_prediction**(model, |                                  |
| > processed_data,                |                                  |
| > classes=\[\'A\', \'B\', \'C\', |                                  |
| > \'D\', \'E\'\]):               |                                  |
+----------------------------------+----------------------------------+
| > probs =                        |                                  |
| > mo                             |                                  |
| del.predict(processed_data)\[0\] |                                  |
+----------------------------------+----------------------------------+
| > **return**                     |                                  |
| > classes\[np.argmax(probs)\]    |                                  |
+----------------------------------+----------------------------------+
| ●                                | > Run the model                  |
+----------------------------------+----------------------------------+
| > input_text = input(\'Enter     |                                  |
| > sentence: \')                  |                                  |
+----------------------------------+----------------------------------+
| > processed_data =               |                                  |
| > prepare_data(input_text,       |                                  |
| > tokenizer)                     |                                  |
+----------------------------------+----------------------------------+
| > result =                       |                                  |
| >                                |                                  |
| make_prediction(sentiment_model, |                                  |
| > processed_data=processed_data) |                                  |
+----------------------------------+----------------------------------+
| > print(f\"Predicted Class:      |                                  |
| > {result}\")                    |                                  |
+----------------------------------+----------------------------------+

> \'\'\'

+-----------------------------------------------------+
| > \^ This is the output of the previous line \^     |
+=====================================================+
| > \'\'\'                                            |
+-----------------------------------------------------+
| > Enter a sentence: **\"This belongs to B Class\"** |
+-----------------------------------------------------+
| > **Predicted Class: B**                            |
+-----------------------------------------------------+
| > \'\'\'                                            |
+-----------------------------------------------------+
| > \^ RESULT \^                                      |
+-----------------------------------------------------+
| > \'\'\'                                            |
+-----------------------------------------------------+

16

> Multi-Label Text Classiﬁcation
>
> Multi-Label Text Classiﬁcation is very similar to the [Multi-Class
> Text Classiﬁcation]{.ul} {page 6} task weelaborated earlier in this
> document. The only difference is that an item (sentence) can belong to
> **one ormore** classes (categories). Therefore, the **one-hot
> encoding** of this task would look like this**:**

![](vertopal_f4c0706f74e74041915764d82d41ebe1/media/image7.png){width="7.083333333333333in"
height="1.8430544619422573in"}

> As you see, A belongs to multiple types here, resulting in A being
> True for more than one class. Suchmodel could provide us with
> alternatives or probabilities for each class, thus, it can be useful
> in casesthat the answer might not always be deﬁnitive.

17

> References

  ----- -------------------------------------------------------------------------------------------------------------
  1\.   1\. A. Fawcett, "Data Science in 5 minutes: What is one hot encoding?," *Educative*. \[Online\]. Available:
  ----- -------------------------------------------------------------------------------------------------------------

> . \[Accessed: 24-Oct-2022\].

  ----- ---------------------------------------------------------------
  2\.   2\. Theartiﬁcialguy, "NLP-with-deep-learning/bert at master ·
  ----- ---------------------------------------------------------------

> theartiﬁcialguy/NLP-with-deep-learning," *GitHub*. \[Online\].
> Available:

. \[Accessed:

> 24-Oct-2022\].
