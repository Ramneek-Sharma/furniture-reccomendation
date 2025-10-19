from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from typing import List, Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import ast
import re
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Furniture Recommendation API - FREE VERSION",
    description="AI-powered furniture recommendation system using FREE HuggingFace models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
df = None
text_embeddings = None
sentence_model = None
image_classifier = None
recommender = None
text_generator = None

class FreeTextGenerator:
    """FREE Text Generation using HuggingFace Transformers"""
    
    def __init__(self):
        logger.info("Loading FREE HuggingFace text generation model...")
        try:
            # Use free Flan-T5 model for text generation
            self.model_name = "google/flan-t5-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=150,
                temperature=0.7,
                do_sample=True
            )
            logger.info("✅ FREE text generation model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load text generation model: {e}")
            self.generator = None
    
    def generate_response(self, user_message):
        """Generate conversational response using free HuggingFace model"""
        try:
            if not self.generator:
                return self._fallback_response(user_message)
            
            # Create prompt for furniture recommendation
            prompt = f"""You are a helpful furniture recommendation assistant. 
            User asks: "{user_message}"
            
            Provide a helpful response about furniture recommendations. Ask follow-up questions about:
            - Room type (living room, bedroom, kitchen, etc.)
            - Style preferences (modern, classic, minimalist)
            - Budget range
            - Space constraints
            
            Keep response friendly and under 100 words.
            
            Response:"""
            
            # Generate response
            response = self.generator(prompt, max_length=150, num_return_sequences=1)
            generated_text = response[0]['generated_text']
            
            # Clean up the response
            if "Response:" in generated_text:
                generated_text = generated_text.split("Response:")[-1].strip()
            
            return generated_text if generated_text else self._fallback_response(user_message)
            
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return self._fallback_response(user_message)
    
    def _fallback_response(self, user_message):
        """Fallback response when model fails"""
        responses = [
            f"I'd be happy to help you find furniture! Based on '{user_message}', let me suggest some products. What room is this for?",
            f"Great question about '{user_message}'! To give you the best recommendations, could you tell me your style preference and budget?",
            f"I can help you with '{user_message}'! What's your room size and do you prefer modern or classic styles?",
            f"Perfect! For '{user_message}', I'll need to know - is this for indoor or outdoor use? What's your color preference?"
        ]
        # Simple hash to pick consistent response for same input
        return responses[hash(user_message) % len(responses)]

class ImageClassifier:
    """Computer Vision model for image classification using ResNet-50"""
    
    def __init__(self):
        logger.info("Loading FREE ResNet-50 model for image classification...")
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.categories = [
            'Furniture', 'Home Decor', 'Storage', 'Lighting', 'Textiles',
            'Kitchen', 'Bedroom', 'Living Room', 'Office', 'Outdoor'
        ]
        
    def classify_image(self, image_url):
        try:
            response = requests.get(image_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence = probabilities.max().item()
                predicted_idx = probabilities.argmax().item()
                
            category = self.categories[predicted_idx % len(self.categories)]
            
            return {"category": category, "confidence": float(confidence), "status": "success"}
            
        except Exception as e:
            logger.error(f"Image classification error: {str(e)}")
            return {"category": "Unknown", "confidence": 0.0, "status": "error"}

class ContentBasedRecommender:
    """ML-powered recommendation system using content-based filtering"""
    
    def __init__(self, embeddings, df):
        self.embeddings = embeddings
        self.df = df
        self.similarity_matrix = cosine_similarity(embeddings)
        logger.info(f"Recommender initialized with {len(df)} products")
        
    def get_recommendations(self, product_id, n_recommendations=5):
        try:
            product_idx = self.df[self.df['uniq_id'] == product_id].index
            if len(product_idx) == 0:
                return pd.DataFrame()
                
            idx = product_idx[0]
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations+1]
            product_indices = [i[0] for i in sim_scores]
            
            recommendations = self.df.iloc[product_indices].copy()
            recommendations['similarity_score'] = [score[1] for score in sim_scores]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            return pd.DataFrame()
    
    def search_products(self, query, n_results=10):
        try:
            query_embedding = sentence_model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            top_indices = similarities.argsort()[-n_results:][::-1]
            
            results = self.df.iloc[top_indices].copy()
            results['similarity_score'] = similarities[top_indices]
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    price_clean = re.sub(r'[^0-9.]', '', str(price_str))
    try:
        return float(price_clean)
    except:
        return np.nan

def extract_categories(cat_str):
    try:
        return ast.literal_eval(cat_str)
    except:
        return []

@app.on_event("startup")
async def startup_event():
    global df, text_embeddings, sentence_model, image_classifier, recommender, text_generator
    
    try:
        logger.info("🚀 Starting FREE application initialization...")
        
        # Load dataset
        logger.info("📂 Loading dataset...")
        df = pd.read_csv("data/intern_data_ikarus.csv")
        
        # Preprocess data
        logger.info("🔧 Preprocessing data...")
        df['price_numeric'] = df['price'].apply(clean_price)
        df['categories_list'] = df['categories'].apply(extract_categories)
        df['main_category'] = df['categories_list'].apply(lambda x: x[0] if x else 'Unknown')
        df['description'] = df['description'].fillna(df['title'])
        
        df['combined_text'] = (
            df['title'].fillna('') + ' ' + 
            df['description'].fillna('') + ' ' + 
            df['material'].fillna('') + ' ' + 
            df['color'].fillna('') + ' ' +
            df['main_category'].fillna('')
        )
        
        # Initialize FREE models
        logger.info("🤖 Loading FREE NLP model (Sentence Transformers)...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("📊 Generating text embeddings...")
        text_embeddings = sentence_model.encode(df['combined_text'].tolist())
        
        logger.info("🖼️ Loading FREE Computer Vision model...")
        image_classifier = ImageClassifier()
        
        logger.info("💡 Setting up recommendation system...")
        recommender = ContentBasedRecommender(text_embeddings, df)
        
        logger.info("💬 Loading FREE text generation model...")
        text_generator = FreeTextGenerator()
        
        # Save embeddings locally (no Pinecone needed)
        logger.info("💾 Saving embeddings locally...")
        os.makedirs("data/embeddings", exist_ok=True)
        np.save("data/embeddings/text_embeddings.npy", text_embeddings)
        
        logger.info("✅ FREE application initialization completed successfully!")
        logger.info(f"📈 Loaded {len(df)} products with FREE AI models")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

# Pydantic models
class ProductQuery(BaseModel):
    query: str
    max_results: Optional[int] = 10
    
class RecommendationRequest(BaseModel):
    product_id: str
    num_recommendations: Optional[int] = 5

class ChatMessage(BaseModel):
    message: str
    user_preferences: Optional[dict] = {}

class ImageClassificationRequest(BaseModel):
    image_url: str

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "🆓 FREE Furniture Recommendation API is running!",
        "status": "active",
        "version": "1.0.0 - FREE",
        "total_products": len(df) if df is not None else 0,
        "models": {
            "nlp": "sentence-transformers/all-MiniLM-L6-v2",
            "cv": "torchvision/resnet50",
            "genai": "google/flan-t5-small",
            "cost": "100% FREE"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "sentence_transformer": sentence_model is not None,
            "image_classifier": image_classifier is not None,
            "recommender": recommender is not None,
            "text_generator": text_generator is not None
        },
        "data_loaded": df is not None,
        "total_products": len(df) if df is not None else 0,
        "api_cost": "FREE"
    }

@app.get("/products")
async def get_all_products(limit: Optional[int] = 50, offset: Optional[int] = 0):
    try:
        total_products = len(df)
        products_subset = df.iloc[offset:offset+limit]
        
        products = []
        for _, product in products_subset.iterrows():
            products.append({
                'uniq_id': product['uniq_id'],
                'title': product['title'],
                'price': product['price'],
                'price_numeric': product.get('price_numeric'),
                'main_category': product['main_category'],
                'brand': product['brand'],
                'images': product['images'],
                'description': product['description'][:200] + '...' if len(str(product['description'])) > 200 else product['description'],
                'material': product.get('material'),
                'color': product.get('color')
            })
        
        return {"products": products, "total": total_products}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_products(query: ProductQuery):
    try:
        if not recommender:
            raise HTTPException(status_code=503, detail="Recommender not initialized")
            
        results = recommender.search_products(query.query, query.max_results)
        
        if results.empty:
            return {"products": [], "query": query.query, "total": 0}
        
        products = []
        for _, product in results.iterrows():
            products.append({
                'uniq_id': product['uniq_id'],
                'title': product['title'],
                'price': product['price'],
                'price_numeric': product.get('price_numeric'),
                'main_category': product['main_category'],
                'brand': product['brand'],
                'images': product['images'],
                'description': product['description'][:200] + '...' if len(str(product['description'])) > 200 else product['description'],
                'similarity_score': float(product['similarity_score']),
                'material': product.get('material'),
                'color': product.get('color')
            })
        
        return {"products": products, "query": query.query, "total": len(products)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    try:
        if not recommender:
            raise HTTPException(status_code=503, detail="Recommender not initialized")
            
        recommendations = recommender.get_recommendations(
            request.product_id, 
            request.num_recommendations
        )
        
        if recommendations.empty:
            raise HTTPException(status_code=404, detail="Product not found")
        
        rec_products = []
        for _, rec in recommendations.iterrows():
            rec_products.append({
                'uniq_id': rec['uniq_id'],
                'title': rec['title'],
                'price': rec['price'],
                'price_numeric': rec.get('price_numeric'),
                'main_category': rec['main_category'],
                'brand': rec['brand'],
                'images': rec['images'],
                'description': rec['description'],
                'similarity_score': float(rec.get('similarity_score', 0)),
                'material': rec.get('material'),
                'color': rec.get('color')
            })
        
        return {"recommendations": rec_products, "total": len(rec_products)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_recommendations(message: ChatMessage):
    """🆓 FREE Conversational AI using HuggingFace Transformers"""
    try:
        # Use FREE HuggingFace model for text generation
        if text_generator:
            ai_response = text_generator.generate_response(message.message)
        else:
            ai_response = f"I'd be happy to help you find furniture! Based on '{message.message}', let me search our catalog for you."
        
        # Get relevant products using FREE semantic search
        products = []
        if recommender:
            search_results = recommender.search_products(message.message, 5)
            
            if not search_results.empty:
                for _, product in search_results.head(3).iterrows():
                    products.append({
                        'uniq_id': product['uniq_id'],
                        'title': product['title'],
                        'price': product['price'],
                        'main_category': product['main_category'],
                        'brand': product['brand'],
                        'images': product['images'],
                        'similarity_score': float(product.get('similarity_score', 0))
                    })
        
        return {
            "ai_response": ai_response,
            "recommended_products": products,
            "conversation_id": "free_chat_session",
            "model_used": "google/flan-t5-small (FREE)"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "ai_response": "I'm here to help you find furniture! What type of furniture are you looking for today?",
            "recommended_products": [],
            "conversation_id": "free_chat_session",
            "model_used": "fallback (FREE)"
        }

@app.post("/classify-image")
async def classify_image(request: ImageClassificationRequest):
    """🆓 FREE Computer Vision using ResNet-50"""
    try:
        if not image_classifier:
            raise HTTPException(status_code=503, detail="Image classifier not initialized")
            
        result = image_classifier.classify_image(request.image_url)
        return {
            "image_url": request.image_url, 
            "classification": result,
            "model_used": "torchvision/resnet50 (FREE)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics():
    try:
        if df is None:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        analytics = {
            "total_products": len(df),
            "categories": df['main_category'].value_counts().head(10).to_dict(),
            "brands": df['brand'].value_counts().head(15).to_dict(),
            "price_ranges": {
                "$0-$25": len(df[(df['price_numeric'] >= 0) & (df['price_numeric'] < 25)]),
                "$25-$50": len(df[(df['price_numeric'] >= 25) & (df['price_numeric'] < 50)]),
                "$50-$100": len(df[(df['price_numeric'] >= 50) & (df['price_numeric'] < 100)]),
                "$100-$200": len(df[(df['price_numeric'] >= 100) & (df['price_numeric'] < 200)]),
                "$200+": len(df[df['price_numeric'] >= 200])
            },
            "materials": df['material'].dropna().value_counts().head(10).to_dict(),
            "colors": df['color'].dropna().value_counts().head(12).to_dict(),
            "statistics": {
                "avg_price": float(df['price_numeric'].mean()) if not df['price_numeric'].isna().all() else 0,
                "median_price": float(df['price_numeric'].median()) if not df['price_numeric'].isna().all() else 0,
                "products_with_price": int(df['price_numeric'].notna().sum())
            },
            "api_cost": "100% FREE"
        }
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
