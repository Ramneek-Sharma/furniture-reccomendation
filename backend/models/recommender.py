class FurnitureRecommendation:
    def __init__(self, furniture_id, name, category, features):
        self.furniture_id = furniture_id
        self.name = name
        self.category = category
        self.features = features

    def __repr__(self):
        return f"FurnitureRecommendation(furniture_id={self.furniture_id}, name='{self.name}', category='{self.category}', features={self.features})"

def get_furniture_recommendations(user_preferences):
    # Placeholder for recommendation logic
    recommendations = []
    # Logic to generate recommendations based on user_preferences
    return recommendations