# Logic-lab
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item matrix (replace with your data)
data = {'user1': [5, 3, 0, 1],
        'user2': [4, 0, 2, 5],
        'user3': [1, 5, 0, 4],
        'user4': [0, 4, 3, 2]}

df = pd.DataFrame(data, index=['item1', 'item2', 'item3', 'item4'])

# Calculate item similarity
item_similarity = cosine_similarity(df.T)

# Create a DataFrame for item similarity
item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)

# Function to get recommendations for a user
def get_recommendations(user, num_recommendations=2):
    similar_users = item_similarity_df[user].sort_values(ascending=False)[1:num_recommendations+1]
    recommended_items = []
    for similar_user in similar_users.index:
         # Get items liked by similar users but not rated by the user
        items_liked_by_similar_user = df.loc[:, similar_user][df.loc[:, similar_user] > 0]
        items_not_liked_by_user = df.loc[:, user][df.loc[:, user] == 0]
        recommendations = list(set(items_liked_by_similar_user.index).intersection(items_not_liked_by_user.index))
        recommended_items.extend(recommendations)
    return recommended_items

# Example usage
user = 'user1'
recommendations = get_recommendations(user)
print(f"Recommendations for {user}: {recommendations}")
