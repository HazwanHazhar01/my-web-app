import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset (replace with your actual file path)
df = pd.read_csv(r'C:\Users\User\Downloads/halal_cleaned_rdataset.csv')

# Extract unique restaurant names and locations
restaurants = df['Restaurant'].unique().tolist()
locations = df['Location'].unique().tolist()

# Function to generate recommendations only from the chosen location, excluding the chosen restaurant
def recommend_restaurants_in_location(location, previous_restaurant_id=None, n_recommendations=5):
    # Create the user-item matrix for all locations (global SVD calculation)
    global_user_item_matrix = df.pivot_table(index='user_id', columns='restaurant_id', values='Rating').fillna(0)
    
    # Apply SVD to the global user-item matrix
    svd = TruncatedSVD(n_components=10)
    global_matrix_reduced = svd.fit_transform(global_user_item_matrix)
    
    # Compute cosine similarity between restaurants globally
    restaurant_factors = svd.components_.T  # Transpose to get restaurant latent factors
    global_similarity_matrix = cosine_similarity(restaurant_factors)
    
    # Filter restaurants to only include those from the chosen location
    location_df = df[df['Location'] == location]  # Filter for the chosen location
    location_user_item_matrix = location_df.pivot_table(index='user_id', columns='restaurant_id', values='Rating').fillna(0)
    
    if previous_restaurant_id is not None and previous_restaurant_id in global_user_item_matrix.columns:
        # Get index of the selected previous restaurant in the global matrix
        previous_restaurant_idx = global_user_item_matrix.columns.get_loc(previous_restaurant_id)
        
        # Get global similarity scores for the selected restaurant
        global_similarity_scores = global_similarity_matrix[previous_restaurant_idx]
        
        # Sort restaurants by similarity score (descending), exclude the previous restaurant itself
        similar_restaurant_indices = np.argsort(global_similarity_scores)[::-1]  # Sort in descending order
        
        # Get the corresponding restaurant IDs
        similar_restaurant_ids = global_user_item_matrix.columns[similar_restaurant_indices]
        
        # Exclude the previous restaurant from recommendations
        similar_restaurant_ids = [rest_id for rest_id in similar_restaurant_ids if rest_id != previous_restaurant_id]
        
        # Filter similar restaurants to only include those in the chosen location
        top_restaurant_ids = [rest_id for rest_id in similar_restaurant_ids if rest_id in location_user_item_matrix.columns]
        
        # Limit to top N recommendations
        top_restaurant_ids = top_restaurant_ids[:n_recommendations]
        
        if len(top_restaurant_ids) == 0:
            st.warning(f"No similar restaurants found in the selected location: {location}.")
            top_restaurant_ids = location_user_item_matrix.columns[:n_recommendations]  # Fallback: top restaurants from the location
    else:
        st.warning("Previous restaurant not found globally. Showing general recommendations from the location.")
        top_restaurant_ids = location_user_item_matrix.columns[:n_recommendations]  # Fallback: top restaurants from the location
    
    # Map restaurant IDs to their names and dietary preferences
    restaurant_info = df[['restaurant_id', 'Restaurant', 'dietary_preference']].drop_duplicates()
    
    # Merge to get restaurant names and dietary preferences for top recommendations
    recommendations = restaurant_info[restaurant_info['restaurant_id'].isin(top_restaurant_ids)]
    
    return recommendations  # Return the recommendations with dietary preference

# Streamlit App
st.title("Restaurant Recommendation System")

# Step 1: User Location Selection
user_location = st.selectbox("Select Your Location:", locations)

# Step 2: Restaurant Selection or ID Input
if 'restaurant_name' not in st.session_state:
    st.session_state.restaurant_name = None

restaurant_selected = st.selectbox(
    "Please select the restaurant you enjoyed the most from the list below. If your favorite is not included, simply choose 'Not Available.' Your selection will help us recommend dining options that are tailored to your preferences!",
    ["Not Available"] + restaurants
)

# Initialize previous_restaurant_id as None
previous_restaurant_id = None

if restaurant_selected == "Not Available":
    st.write("No specific restaurant selected. Recommendations will be general for your selected location.")
else:
    # Fetch restaurant ID from the selected name
    previous_restaurant_id_row = df[df['Restaurant'] == restaurant_selected]
    if not previous_restaurant_id_row.empty:
        previous_restaurant_id = previous_restaurant_id_row['restaurant_id'].values[0]
    else:
        st.error("Selected restaurant not found in dataset.")

# Step 3: Generate Recommendations
if st.button("Get Recommendations"):
    if not user_location:
        st.error("Please select a location.")
    else:
        # Call the recommendation function
        top_restaurants = recommend_restaurants_in_location(user_location, previous_restaurant_id)

        if not top_restaurants.empty:
            st.write(f"Top {len(top_restaurants)} recommended restaurants in {user_location}:")

            # Reset index to remove the default DataFrame index
            top_restaurants_cleaned = top_restaurants[['Restaurant', 'dietary_preference']].reset_index(drop=True)
            
            # Display the cleaned table with Restaurant and Dietary Preference
            st.table(top_restaurants_cleaned)
        else:
            st.write("No recommendations found.")
