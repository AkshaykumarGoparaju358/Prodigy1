import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training_data_path = '/content/twitter_training.csv'
validation_data_path = '/content/twitter_validation.csv'

training_data = pd.read_csv(training_data_path, header=None)
validation_data = pd.read_csv(validation_data_path, header=None)

columns = ['ID', 'Entity', 'Sentiment', 'Tweet']
training_data.columns = columns
validation_data.columns = columns

print("Training Data:")
print(training_data.head())
print("\nValidation Data:")
print(validation_data.head())

combined_data = pd.concat([training_data, validation_data])

plt.figure(figsize=(10, 6))
sns.countplot(data=combined_data, x='Sentiment', order=combined_data['Sentiment'].value_counts().index)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(14, 8))
sns.countplot(data=combined_data, y='Entity', order=combined_data['Entity'].value_counts().index)
plt.title('Entity Distribution')
plt.xlabel('Count')
plt.ylabel('Entity')
plt.show()

plt.figure(figsize=(14, 8))
sns.countplot(data=combined_data, x='Sentiment', hue='Entity', palette='Set2')
plt.title('Sentiment Distribution by Entity')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend(title='Entity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
