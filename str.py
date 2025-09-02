import streamlit as st
import pandas as pd
import joblib
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def get_summary(text):
    if not text.strip():
        return "Rəylər boşdur."
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count=3)
        return " ".join(str(sentence) for sentence in summary)
    except IndexError:
        return "Xülasə çıxarmaq üçün kifayət qədər mətn yoxdur."


model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

df = pd.read_csv('merged_movie_data.csv')

df['review'] = df['review'].fillna('')

st.title('Film Rəyləri və Xülasəsi Analiz Tətbiqi')
st.write('Axtarış panelinə film adını daxil edərək rəylər və xülasəni görə bilərsiniz.')

all_movie_titles = df['title'].unique().tolist()

selected_movie = st.selectbox(
    'Film adını daxil edin:',
    options=all_movie_titles,
    index=None,  
    placeholder='Məsələn: Jurassic World',
    key='movie_search'
)


if selected_movie:
    
    movie_data = df[df['title'] == selected_movie]
    
    if not movie_data.empty:
        st.header(f"Filmin adı: {movie_data['title'].iloc[0]}")
        
        
        all_reviews = ' '.join(movie_data['review'].tolist())
        summary_text = get_summary(all_reviews)
        
        st.subheader("Rəylərdən Çıxarılan Xülasə:")
        st.write(summary_text)

        st.subheader("Seçilmiş Rəylər:")
        reviews = movie_data['review'].sample(n=min(5, len(movie_data['review']))).tolist()
        
        if reviews:
            for review in reviews:
                st.write(f"- {review}")
                
                
                review_vec = vectorizer.transform([review])
                sentiment_pred = model.predict(review_vec)
                sentiment_label = 'Müsbət' if sentiment_pred[0] == 1 else 'Mənfi'
                st.write(f"**Sentiment:** {sentiment_label}")
        else:
            st.write("Bu film üçün rəy tapılmadı.")
    else:
        st.warning("Bu adda film tapılmadı. Zəhmət olmasa başqa film adı daxil edin.")