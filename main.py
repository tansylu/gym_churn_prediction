import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Specify the path to your CSV file
csv_file_path = "gym_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the count of people who churned (Exited the gym)
print(df["Exited"].value_counts())  # Check the count for value "1"

# Exploratory analysis
cols = ['Zipcode', 'Age', 'Partner_company', 'Friend_promo', 'Contract_period',
        'Lifetime', 'Class_registration_weekly', 'Avg_additional_charges_total',
        'Cancellation_freq']
numerical = cols

# # We see that the age distribution is most frequent around 30-40 years old, certain zip codes dominate,
# # and friend promo/partner company are negligible for being 50/50

# # Create a directory for saving boxplot images
# boxplot_dir = "boxplot_images"
# if not os.path.exists(boxplot_dir):
#     os.makedirs(boxplot_dir)

# # Create a directory for saving histogram images
# histogram_dir = "histogram_images"
# if not os.path.exists(histogram_dir):
#     os.makedirs(histogram_dir)

# # Generate histograms for each variable and save them as images
# for col in numerical:
#     plt.figure(figsize=(8, 6))
#     sns.histplot(df[col])
#     plt.title(f"{col} Histogram")
#     plt.savefig(os.path.join(histogram_dir, f"{col}_histogram.png"))  # Save the histogram as an image
#     plt.close()  # Close the plot to release memory

# # Calculate the correlation matrix
# correlation_matrix = df.corr()

# # # Plot the correlation matrix
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.savefig("correlation_matrix.png")  # Save the correlation matrix plot
# plt.show()

# # Based on the correlation matrix, we can see that Age and Cancellation frequency contribute
# # towards a higher chance of a person exiting, while high additional charges and weekly
# # class registration indicate a client not exiting

# # Generate box plots for each variable against "Exited" and save them as images
# for col in df.columns:
#     if col != "Exited":  # Exclude the "Exited" variable itself
#         plt.figure(figsize=(8, 6))
#         sns.boxplot(x="Exited", y=col, data=df)
#         plt.title(f"{col} Boxplot")
#         plt.savefig(os.path.join(boxplot_dir, f"{col}_boxplot.png"))  # Save the boxplot as an image
#         plt.close()  # Close the plot to release memory

# ##Contract period, partner company, registration and friend promo can be omitted as show no diffference


# # Create a directory for saving sns plots
# sns_dir = "sns"
# if not os.path.exists(sns_dir):
#     os.makedirs(sns_dir)

# #Make some count plots for exited and staying customers description
# cols = ['Cancellation_freq', "Avg_additional_charges_total", "Class_registration_weekly", "Contract_period"]

# for col in cols:
#     plt.figure(figsize=(14, 4))
#     ax = sns.countplot(x="Exited", hue=str(col), data=df)
#     plt.title(col)
#     ax.locator_params(axis='x', nbins=5)  # Adjust the number of x-axis ticks
    
#     plt.title(col)
#     plt.savefig(os.path.join(sns_dir, f"{col}_countplot.png"))
#     plt.close()

# #Data preprocessing:

# Display the DataFrame columns info
df = df.drop(df.columns[:3], axis=1)##Drop index, registration and row columns because they dont carry any useful data
df = df.dropna()
df.info()

#the dataset is imbalanced, which means that a majority of values 
# in the target variable belong to a single class.
#  Most customers in the dataset did not churn - only x% of them did.
#>>>solution:
#I will use oversampling


#Before oversampling, doing a train-test split. 
# Will oversample only the training dataset, as the test dataset must be representative of the true population:

X = df.drop(['Exited'],axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

oversample = SMOTE(k_neighbors=5)
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_smote, y_smote

print(y_train.value_counts())

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=46)
rf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score

preds = rf.predict(X_test)
print(accuracy_score(preds,y_test))

# Specify the path to your CSV file
test_file_path = "gym_test.csv"

# Read the CSV file into a DataFrame
df_test = pd.read_csv(test_file_path)
df_test = df_test.drop(df_test.columns[:3], axis=1)##Drop index, registration and row columns because they dont carry any useful data
df_test.info()
print(df_test.head())
X_testing_set = df_test.drop(['Exited'],axis=1)

output = rf.predict(X_testing_set)
df_test["Exited"] = output
df_test.to_excel("output.xlsx")