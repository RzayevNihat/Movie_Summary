# 🎬 Movie Reviews & Summarization Analysis App  

This project provides **sentiment analysis** and **automatic text summarization** based on **IMDB movie reviews**.  
The model is trained using **Logistic Regression**, and an interactive web application is built with **Streamlit**.  

---

## 📌 Features
- Merge IMDB movie list with reviews  
- Perform **positive / negative sentiment analysis** on reviews  
- Generate automatic summaries using **LSA (Latent Semantic Analysis)**  
- Interactive **Streamlit UI** for selecting movies  
- Hyperparameter tuning with **GridSearchCV**  
- Achieved results:  
  - ✅ Validation accuracy: **0.84**  
  - ✅ Test accuracy: **0.84**  

---

## 🛠️ Technologies
- **Python**
- **Pandas, Scikit-learn, Joblib**
- **Sumy (for summarization)**
- **Streamlit (for UI)**

---

## 📂 Project Structure
📦 movie-review-analysis
├── imdb_list.csv # IMDB movie metadata
├── imdb_reviews.csv # IMDB user reviews
├── merged_movie_data.csv # Combined dataset
├── sentiment_model.pkl # Trained sentiment model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── train_model.py # Model training & saving script
├── str.py # Streamlit application
└── README.md # Project documentation


---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/movie-review-analysis.git
cd movie-review-analysis
2️⃣ Install dependencies
pip install -r requirements.txt
requirements.txt should include:
pandas
scikit-learn
joblib
streamlit
sumy
3️⃣ Train the model (if .pkl files are missing)
python train_model.py
4️⃣ Run the Streamlit app
streamlit run str.py
📊 Example Output

After selecting a movie, the app will display:

🎬 Movie title

📝 A 3-sentence summary of reviews

💬 Randomly selected reviews with sentiment labels (Positive / Negative)


✨ Future Improvements

Use Transformer models (BERT, DistilBERT) for more accurate sentiment analysis

Add emotion classification (joy, sadness, anger, etc.)

Visualize sentiment distribution using charts (bar, pie)

Support multi-language reviews

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to improve.

📄 License

This project is licensed under the MIT License – feel free to use and modify.

---

👉 Do you want me to also **generate the `requirements.txt` file** for you right now so you can upload it along with this README?

