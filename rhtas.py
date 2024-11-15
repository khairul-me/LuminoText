import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import nltk
nltk_data_path = "K:/Khairul_Etin_research/tokenizers"
nltk.data.path.append(nltk_data_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import networkx as nx
import spacy
import fitz
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import Counter
from datetime import datetime
import re
import os
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
# Change this import
import community.community_louvain as community_louvain
from typing import List, Dict, Tuple
import matplotlib.dates as mdates

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EnhancedTextAnalyzer:
    def __init__(self, output_path: str):
        """Initialize the enhanced analyzer with all required models."""
        self.output_path = output_path
        self.nlp = spacy.load('en_core_web_md')
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['would', 'could', 'might', 'must', 'page', 'chapter'])
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize T5 for better summarization
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        os.makedirs(output_path, exist_ok=True)

    def process_document(self, pdf_path: str):
        """Main processing pipeline with logging for each step."""
        try:
            # Step 1: Extract and preprocess text
            logging.info("Starting to extract text from the PDF...")
            text = self.extract_text(pdf_path)
            logging.info(f"Text extraction complete. Extracted {len(text)} characters.")

            # Step 2: Clean text
            logging.info("Cleaning the text...")
            cleaned_text = self.clean_text(text)
            logging.info("Text cleaning complete.")

            # Step 3: Generate enhanced summary
            logging.info("Generating enhanced summary...")
            summary = self.generate_enhanced_summary(cleaned_text)
            logging.info("Summary generation complete.")

            # Step 4: Analyze topics
            logging.info("Analyzing topics...")
            topics = self.analyze_topics(cleaned_text)
            logging.info("Topic analysis complete.")

            # Step 5: Analyze sentiment
            logging.info("Analyzing sentiment...")
            sentiment_analysis = self.analyze_sentiment(cleaned_text)
            logging.info("Sentiment analysis complete.")

            # Step 6: Extract timeline events
            logging.info("Extracting timeline events...")
            timeline_events = self.extract_timeline(cleaned_text)
            logging.info("Timeline extraction complete.")

            # Step 7: Create concept network
            logging.info("Creating concept network...")
            concept_network = self.create_concept_network(cleaned_text)
            logging.info("Concept network creation complete.")

            # Step 8: Analyze keyword frequencies
            logging.info("Analyzing keyword frequencies...")
            keyword_frequencies = self.analyze_keywords(cleaned_text)
            logging.info("Keyword frequency analysis complete.")

            # Step 9: Generate visualizations
            logging.info("Generating visualizations...")
            self.create_all_visualizations(
                text=cleaned_text,
                topics=topics,
                sentiment=sentiment_analysis,
                timeline=timeline_events,
                network=concept_network,
                frequencies=keyword_frequencies
            )
            logging.info("Visualizations complete.")

            # Step 10: Save results
            logging.info("Saving results...")
            self.save_results(
                summary=summary,
                topics=topics,
                sentiment=sentiment_analysis,
                timeline=timeline_events,
                frequencies=keyword_frequencies
            )
            logging.info("Results saved successfully.")

        except Exception as e:
            logging.error(f"Error in process_document: {e}")
            raise



    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF with structural preservation."""
        doc = fitz.open(pdf_path)
        text_blocks = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_blocks.append({
                                'text': span['text'],
                                'size': span['size']
                            })
        return ' '.join([block['text'] for block in text_blocks])

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()

    def generate_enhanced_summary(self, text: str) -> str:
        """Generate an enhanced summary with better context and structure."""
        try:
            # Split text into smaller chunks
            chunks = self.split_into_chunks(text, max_chunk_size=500)  # Reduced chunk size
            summaries = []
            
            for chunk in chunks[:3]:  # Process only first 3 chunks
                inputs = self.tokenizer(
                    "summarize: " + chunk,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=150,
                    min_length=50,
                    num_beams=2,
                    early_stopping=True
                )
                
                summaries.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            
            return " ".join(summaries)
            
        except Exception as e:
            logging.error(f"Error in summary generation: {e}")
            return "Summary generation failed: Text too complex or long."

    def split_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            if current_size + len(sentence) > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = len(sentence)
            else:
                current_chunk.append(sentence)
                current_size += len(sentence)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def combine_summaries(self, summaries: List[str]) -> str:
        """Intelligently combine chunk summaries."""
        combined_text = ' '.join(summaries)
        return self.generate_enhanced_summary(combined_text)

    def analyze_topics(self, text: str) -> Dict:
        """Perform topic modeling and analysis."""
        # Convert stop_words set to list
        stop_words_list = list(self.stop_words)
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words_list)
        tfidf = vectorizer.fit_transform([text])
        
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(tfidf)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics[f'Topic_{idx+1}'] = top_words
            
        return topics

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment around key themes."""
        sentences = sent_tokenize(text)
        themes = {
            'race': [],
            'education': [],
            'society': [],
            'politics': []
        }
        
        for sent in sentences:
            scores = self.sia.polarity_scores(sent)
            for theme, keywords in {
                'race': ['race', 'racial', 'black', 'white'],
                'education': ['education', 'university', 'study'],
                'society': ['society', 'social', 'community'],
                'politics': ['politics', 'political', 'government']
            }.items():
                if any(keyword in sent.lower() for keyword in keywords):
                    themes[theme].append(scores)
        
        return {theme: self._average_scores(scores) for theme, scores in themes.items()}

    def _average_scores(self, scores_list: List[Dict]) -> Dict:
        """Calculate average sentiment scores."""
        if not scores_list:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0, 'count': 0}
            
        return {
            'compound': np.mean([s['compound'] for s in scores_list]),
            'pos': np.mean([s['pos'] for s in scores_list]),
            'neg': np.mean([s['neg'] for s in scores_list]),
            'neu': np.mean([s['neu'] for s in scores_list]),
            'count': len(scores_list)
        }

    def extract_timeline(self, text: str) -> List[Dict]:
        """Extract timeline of events."""
        doc = self.nlp(text)
        events = []
        
        for ent in doc.ents:
            if ent.label_ == "DATE":
                sent = ent.sent
                events.append({
                    'date': ent.text,
                    'event': sent.text
                })
                
        return sorted(events, key=lambda x: x['date'])

    def create_concept_network(self, text: str) -> nx.Graph:
        """Create hierarchical concept network."""
        doc = self.nlp(text)
        G = nx.Graph()
        
        # Add nodes and edges based on sentence relationships
        for sent in doc.sents:
            key_words = [token.text.lower() for token in sent 
                        if not token.is_stop and token.pos_ in ['NOUN', 'PROPN']]
                        
            for i in range(len(key_words)):
                for j in range(i + 1, len(key_words)):
                    if G.has_edge(key_words[i], key_words[j]):
                        G[key_words[i]][key_words[j]]['weight'] += 1
                    else:
                        G.add_edge(key_words[i], key_words[j], weight=1)
        
        return G

    def analyze_keywords(self, text: str) -> Dict:
        """Analyze keyword frequencies with additional context."""
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc 
                if not token.is_stop and not token.is_punct and len(token.text) > 2]
        
        return dict(Counter(words).most_common(20))

    def create_all_visualizations(self, **kwargs):
        """Create all visualizations."""
        # Timeline visualization
        self.create_timeline_viz(kwargs['timeline'])
        
        # Topic clusters
        self.create_topic_clusters(kwargs['topics'])
        
        # Enhanced concept network
        self.create_enhanced_network_viz(kwargs['network'])
        
        # Sentiment analysis
        self.create_sentiment_viz(kwargs['sentiment'])
        
        # Keyword frequencies
        self.create_frequency_viz(kwargs['frequencies'])

    def create_timeline_viz(self, events: List[Dict]):
        """Create timeline visualization."""
        plt.figure(figsize=(15, 8))
        
        dates = [event['date'] for event in events]
        events_text = [event['event'] for event in events]
        
        plt.plot(dates, range(len(dates)), 'o-')
        
        for i, txt in enumerate(events_text):
            plt.annotate(txt, (dates[i], i), xytext=(10, 0), 
                        textcoords='offset points')
        
        plt.title('Timeline of Events')
        plt.savefig(os.path.join(self.output_path, 'timeline.png'))
        plt.close()

    def create_topic_clusters(self, topics: Dict):
        """Create topic cluster visualization."""
        plt.figure(figsize=(15, 10))
        
        for i, (topic, words) in enumerate(topics.items()):
            plt.subplot(3, 2, i+1)
            wordcloud = WordCloud(width=400, height=200, background_color='white')
            wordcloud.generate(' '.join(words))
            plt.imshow(wordcloud)
            plt.title(f'Topic {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'topic_clusters.png'))
        plt.close()

    def create_enhanced_network_viz(self, G: nx.Graph):
        """Create enhanced network visualization."""
        plt.figure(figsize=(20, 20))
        
        # Apply community detection
        communities = community_louvain.best_partition(G)
        
        # Calculate node sizes based on degree centrality
        sizes = nx.degree_centrality(G)
        node_sizes = [v * 3000 for v in sizes.values()]
        
        # Draw the network
        pos = nx.spring_layout(G)
        nx.draw(G, pos,
                node_color=list(communities.values()),
                node_size=node_sizes,
                with_labels=True,
                font_size=8,
                width=0.5)
        
        plt.title('Concept Network')
        plt.savefig(os.path.join(self.output_path, 'concept_network.png'))
        plt.close()

    def create_sentiment_viz(self, sentiment: Dict):
        """Create sentiment analysis visualization."""
        plt.figure(figsize=(12, 6))
        
        themes = list(sentiment.keys())
        compounds = [s['compound'] for s in sentiment.values()]
        
        plt.bar(themes, compounds)
        plt.title('Sentiment Analysis by Theme')
        plt.ylabel('Compound Sentiment Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'sentiment_analysis.png'))
        plt.close()

    def create_frequency_viz(self, frequencies: Dict):
        """Create keyword frequency visualization."""
        plt.figure(figsize=(15, 8))
        
        words = list(frequencies.keys())
        counts = list(frequencies.values())
        
        plt.bar(words, counts)
        plt.title('Keyword Frequencies')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'keyword_frequencies.png'))
        plt.close()

    def save_results(self, **kwargs):
        """Save all analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(os.path.join(self.output_path, f'analysis_report_{timestamp}.txt'), 'w') as f:
            f.write("=== Enhanced Analysis Report ===\n\n")
            
            # Summary
            f.write("1. Generated Summary:\n")
            f.write("-" * 50 + "\n")
            f.write(kwargs['summary'] + "\n\n")
            
            # Topics
            f.write("2. Main Topics:\n")
            f.write("-" * 50 + "\n")
            for topic, words in kwargs['topics'].items():
                f.write(f"{topic}: {', '.join(words)}\n")
            f.write("\n")
            
            # Sentiment Analysis
            f.write("3. Sentiment Analysis:\n")
            f.write("-" * 50 + "\n")
            for theme, scores in kwargs['sentiment'].items():
                f.write(f"{theme}:\n")
                for metric, value in scores.items():
                    f.write(f"  {metric}: {value:.3f}\n")
            f.write("\n")
            
            # Timeline
            f.write("4. Key Events Timeline:\n")
            f.write("-" * 50 + "\n")
            for event in kwargs['timeline']:
                f.write(f"{event['date']}: {event['event']}\n")
            f.write("\n")
            
            # Keywords
            f.write("5. Top Keywords and Their Frequencies:\n")
            f.write("-" * 50 + "\n")
            # Keywords section continuation
            for word, freq in kwargs['frequencies'].items():
                f.write(f"{word}: {freq}\n")

def main():
    """Main execution function."""
    try:
        # Define paths
        BASE_PATH = "K:/Khairul_Etin_research"
        PDF_PATH = os.path.join(BASE_PATH, "AppiahChapter04.pdf")
        OUTPUT_PATH = os.path.join(BASE_PATH, "output")
        
        # Initialize analyzer
        analyzer = EnhancedTextAnalyzer(OUTPUT_PATH)
        logging.info("Initialized analyzer successfully")
        
        # Process document
        analyzer.process_document(PDF_PATH)
        logging.info("Document processing completed successfully")
        
        logging.info(f"All outputs have been saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()