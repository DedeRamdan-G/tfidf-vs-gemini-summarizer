import re
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import os
import google.generativeai as genai
from dotenv import load_dotenv
from rouge_score import rouge_scorer
import logging
from .models import Text, Summary, RougeScore, Comparison
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import numpy as np



# Load environment variables from .env file
load_dotenv()

# Configure Gemini AI
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise KeyError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def preprocess_steps(text):
    steps = {}
    # Case folding
    text_lower = text.lower()
    steps['case_folding'] = text_lower
    
    # Menghapus angka dan tanda baca
    text_clean = re.sub(r'\d+', '', text_lower)
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    steps['cleaning'] = text_clean
    
    # Menghapus URL dan email
    text_clean = re.sub(r'http\S+|www\S+|https\S+', '', text_clean, flags=re.MULTILINE)
    text_clean = re.sub(r'\S+@\S+', '', text_clean)
    steps['cleaning'] = text_clean

    # Tokenisasi
    words = nltk.word_tokenize(text_clean)
    steps['tokenizing'] = ' '.join(words)
    
    # Menghapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    words_no_stopwords = [word for word in words if word not in stop_words]
    steps['stopwords_removal'] = ' '.join(words_no_stopwords)
    
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    processed_words = [stemmer.stem(word) for word in words_no_stopwords]
    steps['stemming'] = ' '.join(processed_words)
    
    return steps, ' '.join(processed_words)

def summarize(text, num_sentences=3):
    _, processed_text = preprocess_steps(text)
    sentences = nltk.sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    cosine_similarities = cosine_similarity(X, X)
    sentence_scores = cosine_similarities.sum(axis=1)
    top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    summary = [sentences[i] for i in sorted(top_sentence_indices)]
    
    return ' '.join(summary)

def gemini_summarize(text):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(f"Tolong Ringkas Kalimat Ini: {text}")
    return response.text

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {metric: round(score.fmeasure, 3) for metric, score in scores.items()}


def dashboard(request):
    tfidf_summary = ""
    gemini_summary = ""
    rouge_scores_tfidf = {}
    rouge_scores_gemini = {}
    
    if request.method == 'POST':
        text = request.POST.get('text')
        if text:
            # Save the original text to the database
            original_text = Text(content=text)
            original_text.save()

            tfidf_summary = summarize(text)
            gemini_summary = gemini_summarize(text)
            
            # Save summaries to the database
            tfidf_summary_entry = Summary(text=original_text, method='TF-IDF', summary=tfidf_summary)
            tfidf_summary_entry.save()
            
            gemini_summary_entry = Summary(text=original_text, method='AI', summary=gemini_summary)
            gemini_summary_entry.save()

            # Hitung ROUGE score
            rouge_scores_tfidf = calculate_rouge(text, tfidf_summary)
            rouge_scores_gemini = calculate_rouge(text, gemini_summary)

            # Save ROUGE scores to the database
            RougeScore(summary=tfidf_summary_entry, rouge1=rouge_scores_tfidf['rouge1'],
                       rouge2=rouge_scores_tfidf['rouge2'], rougeL=rouge_scores_tfidf['rougeL']).save()
            RougeScore(summary=gemini_summary_entry, rouge1=rouge_scores_gemini['rouge1'],
                       rouge2=rouge_scores_gemini['rouge2'], rougeL=rouge_scores_gemini['rougeL']).save()

            # Save comparison to the database
            Comparison(text=original_text, tfidf_summary=tfidf_summary_entry, ai_summary=gemini_summary_entry).save()
            
            # Simpan nilai-nilai ke session
            request.session['text'] = text
            request.session['summary'] = tfidf_summary
            request.session['gemini_summary'] = gemini_summary
            request.session['rouge_scores_tfidf'] = rouge_scores_tfidf
            request.session['rouge_scores_gemini'] = rouge_scores_gemini

    return render(request, 'dashboard.html', {
        'tfidf_summary': tfidf_summary,
        'gemini_summary': gemini_summary,
        'rouge_scores_tfidf': rouge_scores_tfidf,
        'rouge_scores_gemini': rouge_scores_gemini
    })

def detail(request):
    text = request.session.get('text', '')
    summary = request.session.get('summary', '')
    gemini_summary = request.session.get('gemini_summary', '')
    rouge_scores_tfidf = request.session.get('rouge_scores_tfidf', {})
    rouge_scores_gemini = request.session.get('rouge_scores_gemini', {})
    preprocessing_steps = {}

    if text:
        preprocessing_steps, _ = preprocess_steps(text)

    return render(request, 'detail.html', {
        'preprocessing_steps': preprocessing_steps,
        'original_text': text,
        'summary': summary,
        'gemini_summary': gemini_summary,
        'rouge_scores_tfidf': rouge_scores_tfidf,
        'rouge_scores_gemini': rouge_scores_gemini
    })

def get_scholar_results(query, num_results=10):
    url = f'https://scholar.google.com/scholar?q={query}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    results = []
    while len(results) < num_results:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        new_results = soup.find_all('div', class_='gs_ri')
        for result in new_results:
            if len(results) >= num_results:
                break
            title = result.find('h3').text
            link = result.find('a')['href']
            snippet = result.find('div', class_='gs_rs').text
            info = result.find('div', class_='gs_a').text

            results.append({
                'Title': title,
                'Link': link,
                'Snippet': snippet,
                'Info': info
            })

        next_button = soup.find('a', string='Next')
        if next_button:
            url = 'https://scholar.google.com' + next_button['href']
        else:
            break

    return results

def scholar_results(request):
    query = request.GET.get('query', '')
    num_results = int(request.GET.get('num_results', 10))
    results = get_scholar_results(query, num_results)

    return render(request, 'scholar_results.html', {
        'results': results,
        'query': query,
        'num_results': num_results
    })
    
def scrape_news(request):
    url = "https://www.liputan6.com/news"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    news_data = []
    for p in soup.find_all("article", {"class": "articles--iridescent-list--item articles--iridescent-list--text-item"}):
        judul_elem = p.find("h4")
        if judul_elem:
            judul_link_elem = judul_elem.find("a")
            judul = judul_link_elem.get_text().strip() if judul_link_elem else "Tidak ada judul"
            link = judul_link_elem['href'] if judul_link_elem else "Tidak ada link"
        else:
            judul = "Tidak ada judul"
            link = "Tidak ada link"
        
        kategori_elem = p.find("a", {"class": "articles--iridescent-list--text-item__category"})
        kategori = kategori_elem.get_text().strip() if kategori_elem else "Tidak ada kategori"
        
        tanggal_elem = p.find("time", {"class": "articles--iridescent-list--text-item__time timeago"})
        tanggal = tanggal_elem.get_text().split(',')[0].strip() if tanggal_elem else "Tidak ada tanggal"
        
        gambar = None
        gambar_elem = p.find("img")
        if gambar_elem:
            gambar = (gambar_elem.get("data-image") or 
                      gambar_elem.get("data-src") or 
                      gambar_elem.get("src"))
        if not gambar:
            gambar = "Tidak ada gambar"
        
        news_data.append({
            'judul': judul,
            'kategori': kategori,
            'tanggal': tanggal,
            'link': link,
            'gambar': gambar
        })

    context = {
        'news_data': news_data
    }
    return render(request, 'news.html', context)

from django.http import JsonResponse

def fetch_news_content(request):
    url = request.GET.get('url')
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extracting image
    image = soup.find('meta', property='og:image')
    image_url = image['content'] if image else ''
    
    # Extracting title
    title = soup.find('meta', property='og:title')
    title_text = title['content'] if title else ''
    
    # Extracting content
    # Adjust the selector based on the actual structure of the website
    content = (soup.find('div', class_='article-content') or
               soup.find('div', class_='article-body') or
               soup.find('div', class_='read-page--content-body') or
               soup.find('div', {'itemprop': 'articleBody'}) or
               soup.find('article') or
               soup.find('div', {'class': 'content'}))
    
    if content:
        paragraphs = content.find_all('p')
        content_text = ' '.join([p.get_text() for p in paragraphs])
    else:
        content_text = 'Content not available'
    
    return JsonResponse({
        'image': image_url,
        'title': title_text,
        'content': content_text
    })
