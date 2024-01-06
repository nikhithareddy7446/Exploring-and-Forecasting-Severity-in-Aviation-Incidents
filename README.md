
**Exploring and Forecasting Severity in Aviation Incidents**


**Abstract**

The skies are filled with marvels of engineering, carrying people and goods across vast distances. However, the inherent risks associated with flight necessitate robust safety measures. A critical element in ensuring safety is the ability to assess and predict the potential severity of aviation incidents. By understanding the factors that influence the course of an incident, we can develop better strategies to mitigate risks and improve outcomes.

In the current study, a machine learning algorithm-based methodology for severity prediction of airplane accidents is proposed. Four of the many elements that affected the airplane disaster take into account a significant correlation. Based on the quantity of aircraft damage and the number of fatalities, prediction categories are created. A realistic and trustworthy prediction of the severity of an aircraft accident is produced by applying algorithms like Support Vector Machine, Random Forest, AdaBoost Classifier, Logistic Regression, and Decision Tree Classifier, and finally using the Voting Ensemble technique to find out the best algorithm.

_ **Key Words** _ **: Airplane crash, safety, prediction, classification, SVM, Random Forest, AdaBoost, Logistic Regression, Decision Tree Classifier, Voting Ensemble Model**

**1. INTRODUCTION**

The increase in technology has resulted in advancements in the systems that are used to predict and analyze existing records. Machine learning is a technology used for automatically making the system learn without explicitly performing the instructions. Large amounts of data concerning records are used for training the model. The attributes that contribute to the crash of a particular airplane are taken into consideration. Classifying or predicting the number of individual data that is collectively associated is called classification. The dataset is filtered and normalized. Predicting the exact classifier class for each record in the dataset is the prime objective of classification. A classifier can tolerate noise and this is its essential quality. Classifiers can handle quantitative data but it is very difficult to carry out this process. Safety is of prime concern for the applications in aviation industries. Companies carry out numerous investigations to create reports and collect information to justify the crash records and hence, this information can be in either conceptual form or structured/non-structured form. Feature selection is one of the prime stages in machine learning. Correct and relatable features should be selected concerning the output to achieve the highest accurate results.

Data contains a lot of redundant values and these values should be filtered out to remove the irrelevant features. Irrelevant values only reduce the valuable assets from the output. The initial number of features is reduced. The new dataset has features that are highly appropriate for predicting the safety aspect of the aircraft [1]. Five algorithms namely Support Vector Machines (SVM), Random Forest, AdaBoost, Logistic Regression, and Decision Tree Classifiers are used for classification. Precision, recall, and f1 score are the performance factors used to improve the accuracy of the classification.

**1.1 Weather Conditions**

Numerous aviation crashes and events are significantly influenced by the weather. According to statistics from the Federal Aviation Authority (FAA), weather is responsible for 70% of all airport delays. While human error is to blame for most aviation incidents and air crashes, the National Transportation Safety Board (NTSB) says that weather is most often a contributing factor. According to their figures, weather is a factor in 23% of all incidents [2]. The entire weather impact accounts for $3 billion in national expenses related to delays, unanticipated maintenance expenditures, and accident damage and injuries. One of the biggest concerns in aviation safety continues to be weather-related general aviation accidents; most of the time, deadly results may be prevented. In addition to safety, convective weather poses a problem for the efficient operation of the NAS. Thunderstorms and related phenomena can close airports, degrade airport capacities for acceptance and departure, and hinder or stop ground operations. Convective hazards en route lead to rerouting and diversions that result in excess operating costs and lost passenger time. Lightning and hail damage can remove aircraft from operations and result in both lost revenues and excess maintenance costs [3]

**1.2 Engine Types**

A few aircraft mishaps occur before takeoff. Roughly 15% of all aviation accidents are mechanical. The good news is that most of those accidents are not caused by engine failure unless someone in your family ends up dead. Aerial engines are classified into two primary categories: reciprocating and jet. A jet engine is a response engine that produces thrust by expelling gas quickly. Jet engines can only run in atmospheric environments because they require the surrounding atmosphere to provide oxygen for burning. There are various types of jet engines, including scramjet, ramjet, pulsejet, turboprop, turbofan, and turbojet. Internal combustion engines known as reciprocating engines, or "piston" engines, use rotating pistons to turn a shaft on which a propeller is fixed to generate thrust. With each fatality, aircraft with reciprocating engine types have the highest accident rate (almost 100%). Turbofan engines are used to power contemporary aircraft. These engines are very dependable and offer many years of trouble-free operation. However, many flight crews felt unprepared to identify engine faults because of the constraints of modeling turbofan engine malfunctions.

**1.3 Phases of the flight**

A total of 64% of fatal incidents occurred during the maneuver, takeoff, first ascent, approach, and landing phases. Although only 17% of a pilot's flight time is spent in these phases, these phases account for 70% of all flight accidents [3]. In comparison to cruising flight, these phases present more angles of attack, a greater chance of distraction, and more chances for control mishandling. Crash rates are higher during takeoff and landing, per Boeing's detailed examination of commercial flight accidents from 1959 to 2017. Approximately 50% of incidents happened during descent and landing between 2007 and 2016. 13% of the deaths were related to takeoff and first climb. Eight minutes or less before landing, or during the first three minutes of departure, is said to be when 80% of all aviation accidents happen.

This is a result of the plane being near the ground at these times. Because there is less time to correct a mistake and the plane does not fully utilize its capabilities until it is in the air and en route to its objective, it is more vulnerable. This is because it's the busiest stage of operations, and if something goes wrong, there's no time or way to fix it. At this point, accidents are likely to become riskier for the passengers involved. While traveling, the taxi stage helps in only 10 percent of accidents.

**1.4 Location**

In other areas of transportation, the Geographic Information System (GIS) is an effective tool for spatial analysis and is often used to study the temporal and spatial characteristics of traffic accidents. Location plays a critical role in aviation accidents, affecting search and rescue operations, investigations, and emergency response. Precise location information facilitates quick and efficient emergency response, accelerating the distribution of resources for trauma care and survivor support.

It is essential for organizing search and rescue efforts and increases the likelihood of finding survivors quickly. Precise location data is also necessary for investigators to gather evidence and reconstruct the sequence of events that preceded an accident, which helps identify the reasons and enhance safety. Finding trends in accident sites over time allows for more focused safety precautions and improved regulations. The geographical context aids in the evaluation of flight paths' dangers and results in tactical modifications for increased safety. To address shared concerns, carry out global safety programs, and advance transparency, international collaboration depends on the exchange of location data. In general, emergency response, investigations, safety improvements, and international collaboration are all impacted by the location of aircraft accidents.

**2. PROBLEM STATEMENT**

In today's world, there are various types of predicting applications used to analyze and provide solutions to future records. The airline industry is advancing day by day. Safety measures are taken in every provided situation by the companies. Also, the risk factors are examined for the prevention of human loss. A single airplane crash can lead to a great loss of human life and property. Numerous factors lead to the airplane crash which are the airplane type, built model, weather conditions, make of the airplane, engine type, phase of the flight, etc. Hence, considering all these factors, based on the details of a particular aircraft the analysis of the airplane crash is carried out. To predict whether the airplane is safe or at a risk the application is built where these functionalities are processed and the safety is predicted.

**3. PROPOSED SYSTEM**

The proposed system allows the person using the system to enter the specifications of the flight in order to know whether the flight is safe or has changes of a crash. Based on the records of various airline companies the analysis and prediction of the given input is carried out. Machine learning is a strong and dependable technology to predict values. Five algorithms are used and based on every dataset the best algorithm would be used in order to predict the value. Every dataset varies majorly and hence, the algorithms used for the classification may also differ. To overcome this problem, the system works accordingly.

**3.1 Feature Selection:**

Redundancy, irrelevant data, noise etc. are removed or none the less reduced to a great extent from a huge dataset having multiple attributes. It comes under the preprocessing step in machine learning. The attributes that add value to the desired output are selected based on the specification of the aviation industry. Every attribute is taken into consideration and its importance is measured by relating it to the output required. The attributes that does not contribute to the result or are of least importance are deleted. The final dataset with the selected features are evaluated to check whether the subset is most relevant for prediction. Also, these attributes are sorted in a specific order from highest to lowest based on its importance on the prediction. As a result, only useful and relevant features are added hence, increasing the accuracy of the prediction.

**3.2 Processing on the Dataset:**

Once the pre-processing is carried out the specific dataset is loaded into the system. In order to carry out the processing the dataset must be structured. The given dataset is cleaned, that is all the missing values are removed by using attribute mean for all samples belonging to the same class also known as aggregation. The dataset is now ready to be loaded. For the purpose of performing machine learning algorithms on this dataset the data has to be split into training data and testing data. The appropriate split ratio for the dataset is 70:30, 70% for training data and 30% for testing data. Now the dataset is ready for the algorithms to carry out its processing.

**3.3 Train and Test the Classifier:**

To make sure the model can adjust to new inputs, its learned knowledge is assessed during the testing phase apart from the training data. In particular, when dealing with imbalanced datasets, the removal of training records guards against overfitting and underfitting and eliminates the need for memory. This stage acts as a crucial checkpoint, pointing out and fixing problems with the model's capacity for generalization. Four algorithms are compared during training and testing to determine which is the most accurate. This most accurate algorithm is then used for real-world predictions, classifying instances as "safe" or "crash" according to patterns learned.

**3.4 Flowchart**

<img width="355" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/7caec564-d712-4006-a8f9-2a8bff2bf0f5">


**Fig.1** : Waterfall Model - Flowchart

**Fig**.1 Simplified overview of a methodology waterfall. The machine learning model is trained on input data gathered from datasets. Once trained, it can be applied to make predictions for other input data.

**4. CLASSIFICATION ALGORITHMS**

**4.1 SVM (Support Vector Machine) :**

Every data item is plotted with its associated value. Classification is carried out in such a manner that it separates the given classes, here each separated area is known as the hyper-plane It is very essential to group the data items in the dataset to their right hyper-plane. This process is known as the identification of the hyper-plane.

**4.2 ADABoost :**

Weak classifiers should be converted into strong classifiers, hence boosting came into picture in machine learning. Weak classifiers are always beneficial than random guesses. As a result, these classifiers prove to be robust and solve the problem of overfitting when applied on large datasets. Hence, the weak ones provide efficient results rather than random values. A single feature is focused upon which has any random kind of threshold applied on it. If the feature is above the threshold than predicted, it belongs to positive otherwise belongs to negative. AdaBoost stands for 'Adaptive Boosting' which transforms weak learners or predictors to strong predictors in order to solve the problem of classification. For classification, below is the final equation:

<img width="309" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/e0d55d5b-3232-4508-a2c5-8a13aea995af">

Here **ht** designates the classifier **t** th weak and t represents its corresponding weight.

**4.3 Random Forest :**

Random forests are a supervised Machine learning algorithm that is widely used in regression and classification problems and produces, even without hyperparameter tuning a great result most of the time. It is perhaps the most used algorithm because of its simplicity. It builds a number of decision trees on different samples and then takes the majority vote if it's a classification problem. At each node, attributes are chosen at random and used to grow the tree. In terms of estimation, wait until the exercise is finished before felling every tree in the forest. As a result, any classifier with weak correlations can generate a robust classifier using a random forest[6].

Every tree in the random forest is built using randomly sampled random vectors that have the same distribution as all the other trees. Because of its great accuracy, the classifier—which was first created for machine learning—has been well-liked in the remote sensing field and has been used for the classification of remotely sensed pictures. Additionally, it meets the process's requirements for efficient parameters and ideal speed. bootstrap random samples for random forest classification, where the best guess across all trees is chosen.

In our scenario, random forest classifiers are used to categorize the seriousness of airplane accidents. To account for the relative importance or influence of each function in the estimate process, Scikit-Learn incorporates an additional model variable. It determines each feature's score automatically throughout the training phase. Thus, until all scores add up to one, relevance is decreased. In order to develop the model, this score aids in identifying the crucial components and eliminating the less important ones. The following is the formula for determining a decision tree's entropy:

<img width="473" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/33a4dace-aabb-41ac-a0e5-f8c669749f4f">

**4.4 Logistic Regression:**

Machine learning approach for binary classification is called logistic regression. It is an easy-to-implement, straightforward method that can serve as a performance benchmark. It is based on data statistics, just as a lot of other machine learning methods. Moreover, despite its name, it is not a collection of guidelines for regression issues where we wish to anticipate an endless outcome. For binary classification, logistic regression is the preferred method. The outcome is a discrete binary between 0 and 1. By utilizing its built-in logistic property to estimate probabilities, logistic regression analyzes the relationship between dependent variables and one or more independent variables.

<img width="241" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/368ba56f-4c6f-4693-a071-67002de6d645">

**4.5 Decision Tree classifier:**

Decision tree methodology is a commonly used data mining method for establishing classification systems based on multiple covariates or for developing prediction algorithms for a target variable. This method classifies a population into branch-like segments that construct an inverted tree with a root node, internal nodes, and leaf nodes. The algorithm is non-parametric and can efficiently deal with large, complicated datasets without imposing a complicated parametric structure. When the sample size is large enough, study data can be divided into training and validation datasets. Using the training dataset to build a decision tree model and a validation dataset to decide on the appropriate tree size needed to achieve the optimal final model [7].

Decision trees are built using a heuristic called recursive partitioning. After the root node, every node is divided into several nodes. The main concept is to divide the data space into dense and sparse regions using a decision tree. Binary or multiway splitting of a binary tree is possible. The tree is kept splitting by the algorithm until the data is somewhat homogeneous. Upon completion of training, an optimal categorized prediction can be made using the decision tree that is returned. The randomness of the dataset will increase with entropy. It is desirable to have a lower entropy while creating a decision tree [8]. The following is the formula for determining a decision tree's entropy:

<img width="284" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/cd323cf2-a1db-4410-bcef-1c30aed25a21">

This metric can further be used to determine the root node of the decision tree and the number of splits to be made. The root node of a decision tree is often referred to as the decision node or the master node.

<img width="401" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/a14359e6-b4f0-48e8-8f25-706c00ac5c17">

**Fig.2** : Decision Tree Model

As the number of splits increases in a decision tree, the time required to build the tree also increases. However, trees with many splits are prone to overfitting, resulting in poor accuracy. This can be managed by deciding an optimal value for the max\_depth parameter. As the value of this parameter increases, the number of splits also increases.

**4.6 Voting Ensemble Model:**

Ensemble techniques are widely used in machine learning to improve model performance by combining predictions from multiple base models. Voting ensemble is a popular method in the realm of ensemble techniques that leverages the wisdom of the crowd by aggregating the outputs of individual models. In machine learning, ensemble techniques combine predictions from multiple base models to boost model performance more than what a single model could achieve. The underlying idea is that different models may capture diverse patterns or biases in the data, and by aggregating their predictions, the ensemble model can compensate for individual model weaknesses and produce better overall performance [8]. Ensembles often achieve higher accuracy than individual models, particularly when the individual models have complementary strengths. Moreover, they are more robust to noise and outliers in the data, as errors made by individual models are often compensated by others.

<img width="322" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/d5aff59c-b161-4e34-8fbd-265b36d9f2e6">

**Fig.3** : Voting Ensemble Model

**5. DATA ANALYSIS**

Companies produce their records for each airplane and hence, these records are collected to form a dataset wherein details about every airplane module are stored hence, a huge dataset comprising thousands of records is formed. Such datasets are loaded with a large amount of attributes. Numerous attributes are required in order to justify the airplane incident. Hence, the data in these attributes is unstructured and textual [9]. The dataset has to be brought and cut down in such a manner where the classification algorithms can be performed. The attributes that define the output of the prediction are the target attributes. The target variables depend upon the safety of the airplane.

<img width="281" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/e21fad01-758c-4d6a-80c3-1e344e911bd5">

**Fig.4** : Data Collection and Analysis

Classification algorithms carry out the process of data analysis. Since accuracy is the main performance measure, the best classification algorithm is selected for the purpose of prediction depending on the dataset. The below table depicts the accuracy of each algorithm.

**6 RESULT**

**Table I.** Result Table

| **Algorithm** | **Accuracy** |
| --- | --- |
| Decision Tree | 88.12% |
| Random Forest | 87.62% |
| Logistic Regression | 87.17% |
| SVM | 87.24% |
| Ada Boost | 87.99% |
| Voting Ensemble | 87.45% |

<img width="481" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/c0499613-2bd1-415b-8536-5ae5e96fb7e1">

**Fig.6** : shows the accuracy of all the algorithms discussed.

The accuracy (in percentage) of each algorithm is displayed in Table 1. Conversely, the Decision Tree classifier has an astounding 88% accuracy rate, which can be attributed to a number of significant characteristics that the algorithm highlights.

<img width="502" alt="image" src="https://github.com/nikhithareddy7446/Exploring-and-Forecasting-Severity-in-Aviation-Incidents/assets/142128157/06fe10fc-e5c8-44b1-9e6b-3b45087caaf5">

**Fig.7** : shows the ROC Curve

**7. CONCLUSIONS**

Five distinct classification algorithms and a voting ensemble model are used in this study to carry out the classification. Every one of the five categories of classification algorithms is tried on the dataset, and each algorithm's patterns are assessed. The correctness of the results mostly determines the system's overall performance, which is crucial. Since the decision tree classifier technique has the highest accuracy, datasets from the aviation sector are typically the ones that use it. The significance of feature selection and how pertinent features impact prediction accuracy have been the main topics of the article.

The dataset has all of its redundant data removed. Therefore, we identify the crucial characteristics that would significantly impact the outcome of the data and sort them in accordance to their ranking. The prediction is helpful for the company and the pilot to take all the necessary steps to avoid an airplane crash. Hence, the classification algorithms have a major role in the data analysis and prediction.

**8. FUTURE SCOPE**

The system is able to predict whether the airplane will be "safe" or not. As a result, the delays of every airplane can also be predicted. The period after which an airplane has to go under the maintenance stage can also be included with the system. Hence, the system will be the one stop destination to check the flight delays, airplane crashes and the period after which the flight should undergo the maintenance phase[10]. The model, with minor modifications, can be integrated into simulation software to better prepare pilots for emergencies and crash prediction systems used to get an idea of the severity of a crash.

**REFERENCES**

| [1] | M. Finlay, "Weather Conditions & Their Airport Impacts Explained," Simple Flying, Jul. 15, 2023. https://simpleflying.com/airport-weather-guide. |
| --- | --- |
| [2] | G. Kulesa, "Weather and Aviation:," 2014. [Online]. Available: https://www.transportation.gov/sites/dot.gov/files/docs/kulesa\_Weather\_Aviation.pdf. |
| [3] | R. Stowell, "Why Do So Many Aviation Accidents Occur During the Maneuvering Phase?," 2010. [Online]. |
| [4] | "NTSB Aviation Investigation Search," _www.ntsb.gov_. https://www.ntsb.gov/Pages/AviationQueryV2.aspx. |
| [5] | Y. Li and C. Liang, "The Analysis of Spatial Pattern and Hotspots of Aviation Accident and Ranking the Potential Risk Airports Based on GIS Platform," _Journal of Advanced Transportation_, vol. 2018, pp. 1–12, Dec. 2018, doi: https://doi.org/10.1155/2018/4027498. |
| [6] | Y.-Y. Song and Y. Lu, "Decision tree methods: applications for classification and prediction," _Shanghai archives of psychiatry_, vol. 27, no. 2, pp. 130–5, 2015, doi: https://doi.org/10.11919/j.issn.1002-0829.215044. |
| [7] | A. Saini, "Random Forest Algorithm for Absolute Beginners in Data Science," _Analytics Vidhya_, Oct. 19, 2021. https://www.analyticsvidhya.com/blog/2021/10/an-introduction-to-random-forest-algorithm-for-beginners |
| [8] | Awan-Ur-Rahman, "Understanding Soft Voting and Hard Voting: A Comparative Analysis of Ensemble Learning Methods," _Medium_, Aug. 02, 2023. https://medium.com/@awanurrahman.cse/understanding-soft-voting-and-hard-voting-a-comparative-analysis-of-ensemble-learning-methods-db0663d2c008#:~:text=Ensemble%20techniques%20are%20widely%20used (accessed Dec. 18, 2023).. |
| [9] | M. J. L. S. M. Aswathy Benn, "Prediction of Aviation Accidents using Logistic Regression Model," 2020. [Online]. Available: https://ijirt.org/master/publishedpaper/IJIRT150433\_PAPER.pdf. |
| [10] | https://www.irjet.net/archives/V7/i3/IRJET-V7I3831.pdf |
| --- | --- |
