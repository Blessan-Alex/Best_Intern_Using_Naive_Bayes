import re
import pandas as pd
import os

# Define the directory containing the chat files
directory_path = '/content/drive/MyDrive/BestInternData/'  # Adjust to your directory

# Initialize a list to hold the cleaned data
cleaned_data = []

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):  # Process only .txt files
        file_path = os.path.join(directory_path, filename)
        
        # Open and read the chat data from the current file
        with open(file_path, 'r', encoding='utf-8') as file:
            chat_data = file.readlines()
        
        # Define the regex pattern for extracting data
        pattern = r'(\d{2}/\d{2}/\d{2}), (\d{1,2}:\d{2}\s?(?:AM|PM|am|pm| [ap]m)) - (.*?): (.*)'
        
        # Extract the relevant data (timestamp, sender, message)
        for line in chat_data:
            match = re.match(pattern, line)
            if match:
                date, time, sender, message = match.groups()
                cleaned_data.append([date, time, sender, message])
            #else:
                #print(f"Line did not match pattern: {line.strip()}")  # Optional: Log unmatched lines

# Convert cleaned data to DataFrame for easier manipulation
chat_df = pd.DataFrame(cleaned_data, columns=['Date', 'Time', 'Sender', 'Message'])

# Display the first few rows of the combined DataFrame
print(chat_df)

# Install nltk and TextBlob
!pip install nltk
!pip install textblob

# Download nltk stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Initialize stop words
stop_words = set(stopwords.words('english'))

# Function to tokenize and remove stopwords
def tokenize_and_clean_message(message):
    # Tokenize the message (split into words)
    words = word_tokenize(message.lower())  # Convert to lowercase for uniformity
    # Remove stopwords
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return words

# Apply tokenization and stopword removal on the cleaned_data
for entry in cleaned_data:
    if entry[2] != 'System Message':  # Skip system messages
        message = entry[3]
        tokens = tokenize_and_clean_message(message)
        entry.append(tokens)  # Add the tokens as a new column


# Function to analyze sentiment using TextBlob
def analyze_sentiment(message):
    blob = TextBlob(message)
    # Polarity ranges from -1 (negative) to 1 (positive)
    polarity = blob.sentiment.polarity
    return polarity

# Apply sentiment analysis
for entry in cleaned_data:
    if entry[2] != 'System Message':  # Skip system messages
        message = entry[3]
        sentiment = analyze_sentiment(message)
        entry.append(sentiment)  # Add sentiment score as a new column
# Print cleaned data with tokens and sentiment
for entry in cleaned_data:
    print(f"Date: {entry[0]}, Time: {entry[1]}, Sender: {entry[2]}")
    print(f"Message: {entry[3]}")
    print(f"Tokens: {entry[4]}")
    print(f"Sentiment Score: {entry[5]}")
    print("-" * 50)
from nltk.probability import FreqDist

# Create a list of all tokens (words) from all messages
all_tokens = []
for entry in cleaned_data:
    if entry[2] != 'System Message':  # Skip system messages
        all_tokens.extend(entry[4])  # The tokens are stored in the 5th column (index 4)

# Calculate frequency distribution
freq_dist = FreqDist(all_tokens)

# Get the 10 most common words
most_common_words = freq_dist.most_common(10)

# Print the most common words with their frequencies
print("Most common words and their frequencies:")
for word, frequency in most_common_words:
    print(f"{word}: {frequency}")
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data for plotting
words = [word for word, freq in most_common_words]
frequencies = [freq for word, freq in most_common_words]

# Create a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=frequencies, y=words, palette="viridis")
plt.title("Top 10 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.show()
# Install wordcloud if not installed
!pip install wordcloud

from wordcloud import WordCloud

# Create word cloud from all tokens
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_tokens))

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
# Extract sentiment scores from the cleaned data
sentiments = [entry[5] for entry in cleaned_data if entry[2] != 'System Message']

# Plot histogram of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(sentiments, kde=True, color="blue")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.show()
import pandas as pd

# Create a DataFrame from cleaned_data
df = pd.DataFrame(cleaned_data, columns=['Date', 'Time', 'Sender', 'Message', 'Tokens', 'Sentiment'])
# Example: Labeling based on a criterion, e.g., positive sentiment
df['Performance'] = df['Sentiment'].apply(lambda x: 'Best Intern' if x > 0.5 else 'Good Intern' if x > 0 else 'Needs Improvement')
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Message'])  # Use messages as features
y = df['Performance']  # Labels
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{cm}')
print(f'Classification Report:\n{report}')
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Create a DataFrame with Sender and Predictions
predictions_df = pd.DataFrame({'Sender': df['Sender'], 'Performance': model.predict(X)})
performance_summary = predictions_df.groupby('Sender')['Performance'].value_counts().unstack(fill_value=0)
print(performance_summary)
best_performers = performance_summary['Best Intern'].idxmax()
best_performer_count = performance_summary['Best Intern'].max()

print(f"The best performer is: {best_performers} with {best_performer_count} votes.")
# Plotting performance distribution
performance_summary.plot(kind='bar', stacked=True)
plt.title('Intern Performance Prediction')
plt.xlabel('Interns')
plt.ylabel('Count of Performance Labels')
plt.xticks(rotation=45)
plt.legend(title='Performance')
plt.tight_layout()
plt.show()
