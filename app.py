# ===============================
# UNIVERSAL AI TUTOR - Streamlit Video Generator
# ===============================
import os
import PyPDF2
import faiss
import numpy as np
import torch
import re
import streamlit as st
import tempfile
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from moviepy import ImageClip, TextClip, CompositeVideoClip, AudioFileClip, ColorClip
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import random
import time
import threading
from queue import Queue

# Token from .env


# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="AURA: AI Video Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    /* Card styling */
    .video-preview-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .script-card {
        background: #1e1e2f;
        border-radius: 15px;
        padding: 1.5rem;
        color: #fff;
        font-family: 'Courier New', monospace;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .script-line {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        background: rgba(255,255,255,0.05);
    }
    .script-time {
        color: #667eea;
        font-weight: bold;
        margin-right: 1rem;
    }
    .timeline-container {
        background: #2d2d3a;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .timeline-item {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
    }
    .generate-btn {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 1rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        font-size: 1.2rem;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    .success-message {
        background: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    .subject-badge {
        display: inline-block;
        padding: 0.2rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .physics { background: #4299e1; color: white; }
    .chemistry { background: #9f7aea; color: white; }
    .biology { background: #48bb78; color: white; }
    .mathematics { background: #ed8936; color: white; }
    .general { background: #718096; color: white; }
    
    /* Progress details */
    .progress-detail {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .progress-percentage {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .progress-status {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Token yahan enter karein
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Use token from .env
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGING_FACE_HUB_TOKEN

# ===============================
# SESSION STATE INITIALIZATION
# ===============================
if 'generated_script' not in st.session_state:
    st.session_state.generated_script = ""
if 'generated_video' not in st.session_state:
    st.session_state.generated_video = None
if 'current_subject' not in st.session_state:
    st.session_state.current_subject = "general"
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'progress_percentage' not in st.session_state:
    st.session_state.progress_percentage = 0
if 'progress_status' not in st.session_state:
    st.session_state.progress_status = "Ready to start"
if 'story_input' not in st.session_state:
    st.session_state.story_input = ""

# ===============================
# 1. FIXED: Use default fonts that exist on all systems
# ===============================
def get_available_font():
    """Get a font that definitely exists on the system"""
    font_options = [
        "Arial", "arial", "DejaVuSans", "Verdana", "Tahoma", 
        "Helvetica", "sans-serif", "LiberationSans"
    ]
    
    # Try common font paths for Windows
    windows_fonts = [
        "C:\\Windows\\Fonts\\Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\Verdana.ttf",
        "C:\\Windows\\Fonts\\Tahoma.ttf"
    ]
    
    for font_path in windows_fonts:
        if os.path.exists(font_path):
            return font_path
    
    # If no font found, let moviepy use default
    return None

# ===============================
# 2. SUBJECT DETECTION
# ===============================
def detect_subject(query):
    """Detect the subject/topic of the question"""
    query_lower = query.lower()
    
    # Physics keywords
    physics_keywords = ['force', 'motion', 'energy', 'gravity', 'velocity', 'acceleration', 
                       'newton', 'physics', 'shm', 'harmonic', 'wave', 'light', 'sound',
                       'magnet', 'electric', 'circuit', 'current', 'voltage', 'resistance',
                       'atom', 'nuclear', 'quantum', 'thermodynamics', 'heat', 'temperature',
                       'lens', 'mirror', 'optics', 'focal', 'convex', 'concave']
    
    # Chemistry keywords
    chemistry_keywords = ['chemical', 'reaction', 'element', 'compound', 'molecule', 'atom',
                         'acid', 'base', 'salt', 'ph', 'periodic', 'table', 'bond', 'organic',
                         'inorganic', 'solution', 'concentration', 'mole', 'stoichiometry',
                         'redox', 'oxidation', 'reduction', 'polymer', 'chemistry']
    
    # Biology keywords
    biology_keywords = ['cell', 'tissue', 'organ', 'system', 'biology', 'life', 'organism',
                       'plant', 'animal', 'human', 'body', 'heart', 'brain', 'lung', 'kidney',
                       'dna', 'rna', 'gene', 'protein', 'enzyme', 'hormone', 'vitamin',
                       'bacteria', 'virus', 'fungus', 'evolution', 'ecosystem', 'food chain',
                       'photosynthesis', 'respiration', 'digestion', 'circulation']
    
    # Mathematics keywords
    math_keywords = ['math', 'algebra', 'calculus', 'geometry', 'trigonometry', 'equation',
                    'function', 'graph', 'derivative', 'integral', 'matrix', 'vector',
                    'probability', 'statistics', 'number', 'theorem', 'proof', 'formula']
    
    # Count matches for each subject
    physics_score = sum(1 for word in physics_keywords if word in query_lower)
    chemistry_score = sum(1 for word in chemistry_keywords if word in query_lower)
    biology_score = sum(1 for word in biology_keywords if word in query_lower)
    math_score = sum(1 for word in math_keywords if word in query_lower)
    
    # Determine subject
    scores = {
        'physics': physics_score,
        'chemistry': chemistry_score,
        'biology': biology_score,
        'mathematics': math_score,
        'general': 1  # Default
    }
    
    subject = max(scores, key=scores.get)
    return subject, max(scores.values())

# ===============================
# 3. LOAD & CHUNK PDF BOOKS
# ===============================
@st.cache_resource
def load_pdfs(folder="book"):
    text = ""
    if not os.path.exists(folder):
        os.makedirs(folder)
        return ""
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            try:
                with open(os.path.join(folder, file), "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        content = page.extract_text()
                        if content: text += content + " "
            except Exception as e:
                st.error(f"Error reading {file}: {e}")
    return text

def chunk_text(text, size=800): 
    return [text[i:i+size] for i in range(0, len(text), size)]

# ===============================
# 4. SMART BULLET POINT FORMATTING (IMPROVED)
# ===============================
def format_as_bullet_points(text, topic, subject):
    """Extract key points and format as bullet points based on subject"""
    
    # Clean the text
    text = text.replace('\n', ' ').strip()
    
    # Subject-specific headers
    subject_headers = {
        'physics': "🔬 PHYSICS EXPLANATION",
        'chemistry': "🧪 CHEMISTRY EXPLANATION", 
        'biology': "🧬 BIOLOGY EXPLANATION",
        'mathematics': "📐 MATHEMATICS EXPLANATION",
        'general': "📚 EXPLANATION"
    }
    
    header = subject_headers.get(subject, "📚 EXPLANATION")
    
    # Start formatting
    formatted = f"► {header}\n"
    formatted += f"► TOPIC: {topic.upper()}\n\n"
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Extract key points (sentences with important keywords)
    key_points = []
    definitions = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 15:  # Meaningful sentences only
            # Separate definitions from regular points
            if any(word in sentence.lower() for word in ['is defined as', 'is called', 'means', 'refers to']):
                definitions.append(f"  • {sentence}")
            elif len(key_points) < 8:  # Max 8 bullet points
                key_points.append(f"  • {sentence}")
    
    # Add definitions first
    if definitions:
        formatted += "► DEFINITION:\n"
        for def_point in definitions[:2]:
            formatted += def_point + "\n"
        formatted += "\n"
    
    # Add key points
    if key_points:
        formatted += "► KEY FEATURES:\n"
        for point in key_points[:5]:  # Limit to 5 points
            # Break long points into multiple lines
            if len(point) > 80:
                words = point.split()
                line = ""
                for word in words:
                    if len(line + word) < 70:
                        line += word + " "
                    else:
                        formatted += line + "\n"
                        line = "  • " + word + " "
                if line:
                    formatted += line + "\n"
            else:
                formatted += point + "\n"
        formatted += "\n"
    
    # Add subject-specific sections
    if subject == 'physics':
        formatted += "► PHYSICAL QUANTITIES:\n"
        # Extract potential formulas
        formulas = re.findall(r'[A-Za-z]+\s*=\s*[A-Za-z0-9^_+\-*/()]+', text)
        for f in set(formulas[:3]):
            formatted += f"  • {f}\n"
    
    elif subject == 'chemistry':
        formatted += "► CHEMICAL FORMULAS:\n"
        # Look for chemical formulas
        chemicals = re.findall(r'[A-Z][a-z]?\d*', text)
        for chem in set(chemicals[:4]):
            if len(chem) < 15:
                formatted += f"  • {chem}\n"
    
    elif subject == 'biology':
        formatted += "► BIOLOGICAL STRUCTURE:\n"
        # Look for biological terms
        bio_terms = re.findall(r'\b[A-Za-z]+\b', text)
        important_terms = [t for t in bio_terms if len(t) > 4 and t[0].isupper()][:4]
        for term in important_terms:
            formatted += f"  • {term}\n"
    
    elif subject == 'mathematics':
        formatted += "► MATHEMATICAL EQUATIONS:\n"
        # Extract equations
        equations = re.findall(r'[A-Za-z0-9\^\_\+\-\*\/\(\)\=]+', text)
        for eq in set(equations[:4]):
            if 3 < len(eq) < 30:
                formatted += f"  • {eq}\n"
    
    # Add examples
    formatted += "\n► EXAMPLES:\n"
    example_sentences = [s for s in sentences if 'example' in s.lower() or 'like' in s.lower() or 'such as' in s.lower()]
    if example_sentences:
        for ex in example_sentences[:2]:
            formatted += f"  • {ex.strip()}\n"
    else:
        formatted += f"  • Common examples of {topic}\n"
        formatted += f"  • Real-life applications\n"
    
    return formatted

# ===============================
# 5. IMPROVED IMAGE GENERATION (BOOK-STYLE DIAGRAMS)
# ===============================
@st.cache_resource
def load_sd_model():
    """Load Stable Diffusion model with caching"""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float32,
            safety_checker=None
        )
        pipe.to("cpu")
        pipe.enable_attention_slicing()
        return pipe
    except Exception as e:
        st.warning(f"Could not load Stable Diffusion: {e}")
        return None

sd_pipe = load_sd_model()

def create_fallback_image(topic, subject):
    """Create a simple diagram when SD fails"""
    # Create a blank image
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # Draw based on subject
    if 'lens' in topic.lower() or 'convex' in topic.lower() or 'concave' in topic.lower():
        # Draw lens diagram
        draw.ellipse([200, 150, 300, 350], outline='black', width=3)
        draw.line([100, 250, 400, 250], fill='black', width=1)
        draw.text((150, 100), f"{topic.title()}", fill='black', font=font)
        
        # Draw arrows for light rays
        draw.line([150, 250, 200, 200], fill='red', width=2)
        draw.line([150, 250, 200, 300], fill='red', width=2)
        draw.polygon([140, 250, 150, 240, 150, 260], fill='red')
        
    elif 'shm' in topic.lower() or 'harmonic' in topic.lower() or 'spring' in topic.lower():
        # Draw spring-mass system
        draw.line([100, 300, 400, 300], fill='black', width=2)
        # Draw spring (zigzag)
        x = 200
        for i in range(8):
            draw.line([x, 300, x+15, 280], fill='black', width=2)
            draw.line([x+15, 280, x+30, 300], fill='black', width=2)
            x += 30
        draw.rectangle([400, 280, 450, 320], outline='black', width=2)
        draw.text((150, 200), "Spring-Mass System", fill='black', font=font)
        draw.text((410, 250), "m", fill='black', font=font)
        
    elif 'circuit' in topic.lower() or 'current' in topic.lower():
        # Draw simple circuit
        draw.rectangle([150, 200, 250, 300], outline='black', width=2)
        draw.text((180, 240), "R", fill='black', font=font)
        draw.line([250, 250, 350, 250], fill='black', width=2)
        draw.ellipse([350, 230, 400, 270], outline='black', width=2)
        draw.text((360, 235), "+", fill='black', font=font)
        draw.line([375, 250, 450, 250], fill='black', width=2)
        draw.text((200, 350), "Simple Circuit", fill='black', font=font)
        
    elif 'cell' in topic.lower() or 'biology' in subject:
        # Draw cell diagram
        draw.ellipse([150, 150, 350, 350], outline='black', width=3)
        draw.ellipse([220, 220, 280, 280], outline='black', width=2)  # nucleus
        draw.text((230, 240), "N", fill='black', font=font)
        # Add some dots for cytoplasm
        for i in range(10):
            x = random.randint(170, 330)
            y = random.randint(170, 330)
            if not (220 < x < 280 and 220 < y < 280):
                draw.point((x, y), fill='black')
        draw.text((200, 370), f"{topic[:20]} Cell", fill='black', font=font)
        
    else:
        # Generic diagram
        draw.rectangle([150, 150, 350, 350], outline='black', width=3)
        draw.text((200, 400), topic[:20], fill='black', font=font)
        draw.text((200, 100), f"{subject.title()} Diagram", fill='black', font=font)
        
        # Add some labels
        draw.text((160, 180), "A", fill='blue', font=font)
        draw.text((300, 280), "B", fill='blue', font=font)
        draw.line([170, 190, 290, 290], fill='red', width=1)
    
    # Save to temp file
    temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    img.save(temp_img.name)
    return temp_img.name

def generate_image(topic, subject, progress_queue=None):
    """Generate textbook-style diagram with progress updates"""
    
    if progress_queue:
        progress_queue.put(("image", 70, "Creating diagram..."))
    
    topic_lower = topic.lower()
    
    # If no SD model, create a simple diagram with PIL
    if sd_pipe is None:
        return create_fallback_image(topic, subject)
    
    # Book-style prompts for clear diagrams
    book_prompts = {
        'physics': f"textbook physics diagram of {topic}, simple clear line drawing, labeled parts, educational illustration, 2D diagram, black and white with minimal colors, scientific journal style, high quality, sharp lines, detailed schematic",
        
        'chemistry': f"textbook chemistry diagram of {topic}, molecular structure or reaction diagram, clear labeled illustration, educational, scientific journal style, 2D schematic, black and white with minimal colors",
        
        'biology': f"textbook biology diagram of {topic}, detailed anatomical drawing, labeled parts, educational illustration, scientific journal style, clear line art, black and white with minimal shading",
        
        'mathematics': f"textbook mathematics diagram of {topic}, geometric shapes or graphs, clear labeled illustration, educational, scientific journal style, 2D diagram, black and white"
    }
    
    prompt = book_prompts.get(subject, 
        f"textbook educational diagram of {topic}, simple clear line drawing, labeled, scientific illustration, 2D, black and white")
    
    # Negative prompt to avoid bad quality
    negative_prompt = "photograph, 3d render, painting, sketch, blurry, low quality, distorted, ugly, bad anatomy"
    
    try:
        image = sd_pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # Save to temp file
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        image.save(temp_img.name)
        
        if progress_queue:
            progress_queue.put(("image", 80, "Diagram created successfully!"))
            
        return temp_img.name
    except Exception as e:
        st.warning(f"Image generation failed: {e}")
        return create_fallback_image(topic, subject)

# ===============================
# 6. VIDEO CREATION WITH PROGRESS
# ===============================
def make_video(image_path, raw_text, question, subject, output_filename, progress_queue=None):
    try:
        # Format text as bullet points
        if progress_queue:
            progress_queue.put(("video", 85, "Formatting text for video..."))
            
        display_text = format_as_bullet_points(raw_text, question, subject)
        
        # Store in session state for display
        st.session_state.generated_script = display_text
        
        # Audio Generation
        if progress_queue:
            progress_queue.put(("video", 90, "Generating audio narration..."))
            
        audio_path = "temp_audio.mp3"
        tts = gTTS(text=raw_text[:800], lang='en')
        tts.save(audio_path)
        audio_clip = AudioFileClip(audio_path)
        duration = max(audio_clip.duration, 10)

        # Background Settings
        bg_colors = {
            'physics': (10, 10, 30),
            'chemistry': (20, 10, 20),
            'biology': (10, 25, 15),
            'mathematics': (25, 15, 10),
            'general': (10, 10, 15)
        }
        bg_color = bg_colors.get(subject, (10, 10, 15))
        
        W, H = 1280, 720
        bg = ColorClip(size=(W, H), color=bg_color).with_duration(duration)

        # Subject Emoji
        subject_emojis = {
            'physics': '🔬',
            'chemistry': '🧪',
            'biology': '🧬',
            'mathematics': '📐',
            'general': '📚'
        }
        emoji = subject_emojis.get(subject, '📚')
        
        # Heading
        heading = TextClip(
            text=f"{emoji} {question.upper()}",
            font_size=42,
            color='white',
            size=(1100, 80),
            method='caption',
            stroke_color='cyan',
            stroke_width=1
        ).with_duration(duration).with_position(("center", 30))

        # Subject label
        subject_label = TextClip(
            text=f"Subject: {subject.title()}",
            font_size=20,
            color='lightgray',
            size=(300, 30),
            method='caption'
        ).with_duration(duration).with_position((50, 650))

        # Left Side Text - Bullet Points
        txt_clip = TextClip(
            text=display_text, 
            font_size=22,
            color='yellow', 
            size=(580, 520),
            method='caption',
            text_align='left'
        ).with_duration(duration).with_position((50, 120))

        # Right Side Diagram
        try:
            img_clip = ImageClip(image_path).with_duration(duration)
            img_clip = img_clip.resized(height=400).with_position((700, 150))
        except:
            img_clip = ColorClip(size=(400, 400), color=(50, 50, 50)).with_duration(duration)
            img_clip = img_clip.with_position((700, 150))

        # Footer
        footer = TextClip(
            text=f"{emoji} {subject.title()} Tutorial • Learn with AI | AURA STUDIO",
            font_size=18,
            color='lightblue',
            size=(600, 30),
            method='caption'
        ).with_duration(duration).with_position(("center", 680))
        
        # Final Assembly
        if progress_queue:
            progress_queue.put(("video", 95, "Rendering video (this may take a moment)..."))
            
        final_video = CompositeVideoClip([bg, heading, subject_label, txt_clip, img_clip, footer])
        final_video = final_video.with_audio(audio_clip)
        
        # Save video
        output_path = f"{output_filename}.mp4"
        final_video.write_videofile(output_path, fps=24, codec="libx264", logger=None)
        
        # Cleanup
        audio_clip.close()
        if os.path.exists(audio_path): 
            os.remove(audio_path)
        
        if progress_queue:
            progress_queue.put(("video", 100, "Video generated successfully!"))
            
        return output_path
        
    except Exception as e:
        st.error(f"Video Creation Error: {e}")
        return None

# ===============================
# 7. UNIVERSAL AI RESPONSE
# ===============================
@st.cache_resource
def initialize_rag():
    """Initialize RAG system"""
    book_data = load_pdfs()
    if not book_data:
        book_data = """
        Physics: Simple Harmonic Motion is periodic motion. Force is proportional to displacement. Convex lens converges light rays. It is thick in middle and thin at edges.
        Chemistry: Chemical reactions involve breaking and forming bonds. Atoms combine to form molecules.
        Biology: Cells are basic units of life. DNA contains genetic information.
        Mathematics: Algebra deals with equations and variables. Calculus studies rates of change.
        """

    chunks = chunk_text(book_data)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    
    return embed_model, index, chunks, t5_tokenizer, t5_model

embed_model, index, chunks, t5_tokenizer, t5_model = initialize_rag()

def get_detailed_answer(query, subject, progress_queue=None):
    """Get detailed answer based on subject"""
    
    if progress_queue:
        progress_queue.put(("rag", 30, "Searching knowledge base..."))
        
    query_vec = embed_model.encode([query])
    _, I = index.search(np.array(query_vec), k=3)
    context = " ".join([chunks[i] for i in I[0]])

    # Subject-specific prompt
    subject_prompts = {
        'physics': "Explain this physics concept in simple terms with formulas and real examples. Include definitions and key principles.",
        'chemistry': "Explain this chemistry concept with chemical equations and everyday examples. Include molecular structures if relevant.",
        'biology': "Explain this biology concept with structures, functions, and examples. Include cellular or anatomical details.",
        'mathematics': "Explain this mathematical concept with equations and step-by-step examples. Include formulas and applications.",
        'general': "Explain this concept clearly with simple examples and real-world applications."
    }
    
    subject_instruction = subject_prompts.get(subject, "Explain this concept clearly with examples.")
    
    prompt = (f"Question: {query}\n"
              f"Subject: {subject}\n"
              f"Instructions: {subject_instruction}\n"
              f"Context: {context[:1500]}\n\n"
              f"Provide a detailed, educational answer with examples:")
    
    if progress_queue:
        progress_queue.put(("rag", 50, "Generating explanation..."))
    
    inputs = t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = t5_model.generate(
        **inputs, 
        max_length=450, 
        min_length=150, 
        num_beams=4,
        repetition_penalty=2.0,
        early_stopping=True
    )
    
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if progress_queue:
        progress_queue.put(("rag", 60, "Explanation ready!"))
    
    return answer

# ===============================
# 8. VIDEO GENERATION THREAD
# ===============================
def generate_video_thread(story, progress_queue):
    """Run video generation in a separate thread with progress updates"""
    try:
        # Step 1: Detect subject
        progress_queue.put(("detect", 10, "Analyzing topic..."))
        subject, confidence = detect_subject(story)
        
        # Step 2: Get answer
        progress_queue.put(("rag", 25, f"Subject detected: {subject.title()}"))
        answer = get_detailed_answer(story, subject, progress_queue)
        
        # Step 3: Format script
        progress_queue.put(("format", 65, "Formatting script..."))
        formatted_script = format_as_bullet_points(answer, story, subject)
        
        # Store script in session state (need to use st.session_state from main thread)
        # We'll handle this in the main loop
        
        # Step 4: Generate image
        progress_queue.put(("image", 70, "Creating diagram..."))
        img_path = generate_image(story, subject, progress_queue)
        
        # Step 5: Generate video
        progress_queue.put(("video", 85, "Rendering video..."))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_file = f"video_{timestamp}"
        
        output_path = make_video(img_path, answer, story, subject, video_file, progress_queue)
        
        if output_path and os.path.exists(output_path):
            progress_queue.put(("complete", 100, "✅ Video generated successfully!", output_path, formatted_script))
        
        # Clean up temp files
        if os.path.exists(img_path):
            os.unlink(img_path)
            
    except Exception as e:
        progress_queue.put(("error", 0, f"Error: {str(e)}"))

# ===============================
# 9. STREAMLIT UI
# ===============================
def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 AURA: AI VIDEO STUDIO</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your ideas into educational videos with AI</p>', unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 Input & Settings")
        
        # Story input - store in session state
        story = st.text_area(
            "Type your story or prompt here...",
            height=150,
            placeholder="e.g., Explain convex lens with diagram, Simple Harmonic Motion, DNA structure...",
            key="story_input"
        )
        
        # Settings row
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=["16:9", "9:16"],
                index=0
            )
        
        with settings_col2:
            style = st.selectbox(
                "Style",
                options=["3D", "Cinematic", "Anime", "Educational"],
                index=3
            )
        
        # Generate button
        if st.button("🚀 GENERATE VIDEO", use_container_width=True, type="primary"):
            if story:
                st.session_state.processing = True
                st.session_state.generated_script = ""
                st.session_state.generated_video = None
                st.session_state.progress_percentage = 0
                st.session_state.progress_status = "Starting generation..."
                st.rerun()
            else:
                st.warning("Please enter a story or prompt first!")
        
        # Progress display (only show when processing)
        if st.session_state.processing:
            st.markdown("### 📊 Generation Progress")
            
            # Progress bar
            progress_bar = st.progress(st.session_state.progress_percentage / 100, 
                                      text=st.session_state.progress_status)
            
            # Percentage display
            st.markdown(f"""
            <div class="progress-detail">
                <div class="progress-percentage">{st.session_state.progress_percentage}%</div>
                <div class="progress-status">{st.session_state.progress_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Timeline preview
        st.markdown("### ⏱️ Timeline")
        timeline_html = """
        <div class="timeline-container">
            <span class="timeline-item">0:00</span>
            <span class="timeline-item">0:05</span>
            <span class="timeline-item">0:12</span>
            <span class="timeline-item">0:15</span>
            <span class="timeline-item">0:20</span>
            <span class="timeline-item">0:35</span>
        </div>
        """
        st.markdown(timeline_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📜 Generated Script")
        
        # Show script if available
        if st.session_state.generated_script:
            # Format script with timeline
            script_lines = st.session_state.generated_script.split('\n')
            script_html = '<div class="script-card">'
            
            # Add timeline markers
            times = ["0:00", "0:05", "0:12", "0:15", "0:20", "0:35"]
            for i, line in enumerate(script_lines[:12]):  # Show more lines
                if line.strip():
                    time_index = i % len(times)
                    script_html += f'<div class="script-line"><span class="script-time">{times[time_index]}</span> {line}</div>'
            
            script_html += f'<div style="margin-top: 1rem; color: #667eea;">⏱️ Script ready for video generation</div>'
            script_html += '</div>'
            st.markdown(script_html, unsafe_allow_html=True)
        else:
            # Default script
            default_script = """
            <div class="script-card">
                <div class="script-line"><span class="script-time">0:00</span> ► EXPLANATION</div>
                <div class="script-line"><span class="script-time">0:05</span> ► TOPIC: Your topic will appear here</div>
                <div class="script-line"><span class="script-time">0:12</span> ► DEFINITION:</div>
                <div class="script-line"><span class="script-time">0:15</span>   • Key points will be listed here</div>
                <div class="script-line"><span class="script-time">0:20</span>   • Important concepts explained</div>
                <div class="script-line"><span class="script-time">0:35</span>   • Examples and applications</div>
                <div style="margin-top: 1rem; color: #667eea;">⏱️ Enter a prompt and click GENERATE</div>
            </div>
            """
            st.markdown(default_script, unsafe_allow_html=True)
        
        st.markdown("### 🎥 Video Preview")
        
        # Video preview placeholder
        if st.session_state.generated_video and os.path.exists(st.session_state.generated_video):
            # Show generated video
            video_file = open(st.session_state.generated_video, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
            # Download button
            with open(st.session_state.generated_video, 'rb') as f:
                st.download_button(
                    label="📥 Download Video",
                    data=f,
                    file_name=os.path.basename(st.session_state.generated_video),
                    mime="video/mp4",
                    use_container_width=True
                )
        elif st.session_state.processing:
            # Show processing animation
            preview_html = f"""
            <div class="video-preview-card">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
                <h3>Generating your video...</h3>
                <p>{st.session_state.progress_status}</p>
                <div style="width: 100%; height: 10px; background: rgba(255,255,255,0.2); border-radius: 10px; margin-top: 2rem;">
                    <div style="width: {st.session_state.progress_percentage}%; height: 100%; background: white; border-radius: 10px; transition: width 0.3s;"></div>
                </div>
                <p style="margin-top: 1rem; font-size: 2rem;">{st.session_state.progress_percentage}%</p>
            </div>
            """
            st.markdown(preview_html, unsafe_allow_html=True)
        else:
            # Show placeholder
            preview_html = """
            <div class="video-preview-card">
                <div style="font-size: 5rem; margin-bottom: 1rem;">🎥</div>
                <h3>Your video preview will appear here</h3>
                <p>Enter your prompt and click GENERATE to start</p>
            </div>
            """
            st.markdown(preview_html, unsafe_allow_html=True)
    
    # Processing logic with thread
    if st.session_state.processing and st.session_state.story_input:
        
        # Create a queue for progress updates
        if 'progress_queue' not in st.session_state:
            st.session_state.progress_queue = Queue()
            st.session_state.thread_started = False
        
        # Start the thread if not already started
        if not st.session_state.get('thread_started', False):
            thread = threading.Thread(
                target=generate_video_thread,
                args=(st.session_state.story_input, st.session_state.progress_queue)
            )
            thread.daemon = True
            thread.start()
            st.session_state.thread_started = True
            st.session_state.thread = thread
        
        # Check for progress updates
        while not st.session_state.progress_queue.empty():
            try:
                update = st.session_state.progress_queue.get_nowait()
                
                if update[0] == "detect":
                    st.session_state.progress_percentage = update[1]
                    st.session_state.progress_status = update[2]
                
                elif update[0] == "rag":
                    st.session_state.progress_percentage = update[1]
                    st.session_state.progress_status = update[2]
                
                elif update[0] == "format":
                    st.session_state.progress_percentage = update[1]
                    st.session_state.progress_status = update[2]
                
                elif update[0] == "image":
                    st.session_state.progress_percentage = update[1]
                    st.session_state.progress_status = update[2]
                
                elif update[0] == "video":
                    st.session_state.progress_percentage = update[1]
                    st.session_state.progress_status = update[2]
                
                elif update[0] == "complete":
                    st.session_state.progress_percentage = update[1]
                    st.session_state.progress_status = update[2]
                    st.session_state.generated_video = update[3]
                    st.session_state.generated_script = update[4]
                    st.session_state.processing = False
                    st.session_state.thread_started = False
                    st.rerun()
                
                elif update[0] == "error":
                    st.session_state.progress_percentage = 0
                    st.session_state.progress_status = update[2]
                    st.session_state.processing = False
                    st.session_state.thread_started = False
                    st.error(update[2])
                    st.rerun()
                    
            except Exception as e:
                pass
        
        # Force a rerun to update UI
        time.sleep(0.5)
        st.rerun()

if __name__ == "__main__":
    main()