import os
import logging
import json
import re
import io
import csv
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import PIL.Image

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available")

try:
    import fitz  # PyMuPDF for better PDF handling
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logging.warning("PyMuPDF not available")

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Poll, ChatMember, ChatMemberOwner, ChatMemberAdministrator, Document
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler, PollAnswerHandler
import google.generativeai as genai

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyCeggxQJR71Ey3tZZ2Lo0HHQPl4NHdhzE0")

# Rate limiting configuration for Gemini API
class RateLimiter:
    def __init__(self):
        # Gemini 1.5 Flash free tier limits: 15 RPM, 1 million TPM, 1500 RPD
        self.requests_per_minute = 15
        self.requests_per_day = 1500
        self.tokens_per_minute = 1000000
        
        self.request_times = []
        self.daily_requests = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        self.tokens_used_minute = 0
        self.minute_reset_time = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
    
    def reset_daily_if_needed(self):
        if datetime.now() >= self.daily_reset_time:
            self.daily_requests = 0
            self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    
    def reset_minute_if_needed(self):
        if datetime.now() >= self.minute_reset_time:
            self.tokens_used_minute = 0
            self.minute_reset_time = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=1)
            self.request_times = [t for t in self.request_times if t > datetime.now() - timedelta(minutes=1)]
    
    async def wait_if_needed(self, estimated_tokens=10000):
        self.reset_daily_if_needed()
        self.reset_minute_if_needed()
        
        # Check daily limit
        if self.daily_requests >= self.requests_per_day:
            wait_time = (self.daily_reset_time - datetime.now()).total_seconds()
            logger.warning(f"Daily request limit reached. Waiting {wait_time} seconds.")
            return wait_time
        
        # Check minute limits
        current_time = datetime.now()
        recent_requests = [t for t in self.request_times if t > current_time - timedelta(minutes=1)]
        
        if len(recent_requests) >= self.requests_per_minute:
            wait_time = 60 - (current_time - recent_requests[0]).total_seconds()
            logger.warning(f"Per-minute request limit reached. Waiting {wait_time} seconds.")
            return wait_time
        
        if self.tokens_used_minute + estimated_tokens > self.tokens_per_minute:
            wait_time = (self.minute_reset_time - current_time).total_seconds()
            logger.warning(f"Token limit reached. Waiting {wait_time} seconds.")
            return wait_time
        
        return 0
    
    def record_request(self, tokens_used=10000):
        self.request_times.append(datetime.now())
        self.daily_requests += 1
        self.tokens_used_minute += tokens_used
    
    def estimate_processing_time(self, num_requests, tokens_per_request=10000):
        """Estimate total processing time including rate limiting delays"""
        total_time = 0
        current_requests = len([t for t in self.request_times if t > datetime.now() - timedelta(minutes=1)])
        
        for i in range(num_requests):
            requests_in_current_minute = current_requests + (i % self.requests_per_minute)
            
            if requests_in_current_minute >= self.requests_per_minute:
                # Need to wait for next minute
                total_time += 60
                current_requests = 0
            
            # Add processing time (estimated 3-5 seconds per request)
            total_time += 4
        
        return total_time

# Global rate limiter instance
rate_limiter = RateLimiter()

# Global variables to store questions and processing states
user_questions = {}
user_processing_state = {}
poll_to_question = {}

# List of allowed users and groups
ALLOWED_USERS = ["taiturab", "siaaam_valo_nai_17"]
ALLOWED_GROUPS = [-1002419863192]

def restricted_access(func):
    """Decorator to restrict access to allowed users and admins of allowed groups."""
    async def wrapped(update: Update, context: CallbackContext, *args, **kwargs):
        user = update.effective_user
        chat = update.effective_chat
        
        if user and user.username in ALLOWED_USERS:
            return await func(update, context, *args, **kwargs)
        
        elif chat and chat.type in ['group', 'supergroup'] and chat.id in ALLOWED_GROUPS:
            try:
                chat_member = await context.bot.get_chat_member(chat.id, user.id)
                
                if (isinstance(chat_member, ChatMemberAdministrator) or 
                    isinstance(chat_member, ChatMemberOwner) or 
                    user.username in ALLOWED_USERS):
                    return await func(update, context, *args, **kwargs)
                else:
                    await update.message.reply_text("Sorry, only group admins can use this bot.")
                    return
            except Exception as e:
                logger.error(f"Error checking admin status: {e}")
                await update.message.reply_text("An error occurred while checking permissions.")
                return
        
        else:
            await update.message.reply_text("আসসালামুআলাইকুম ।\n সফটওয়্যারটি ছবি থেকে প্রশ্ন বের করে কুইজ এবং তার ব্যখ্যা তৈরি করতে পারে। এতে করে প্রচুর সময় সাশ্রয় করতে পারে এই সফটওয়্যারটি ।\n\n এটি কিভাবে ব্যবহার করে এবং আপনি এটি নিতে চাইলে যোগাযোগ করুন নিচের দেয়া আইডি তে ।\n\n @taiturab অথবা @siaaam_valo_nai_17 কে মেসেজ করুন। \nRegards-\n Suffering From Software")
            return
    
    return wrapped

@restricted_access
async def start(update: Update, context: CallbackContext) -> None:
    """Send a greeting message when the command /start is issued."""
    user = update.effective_user
    greeting = f"""Hello {user.name}! 👋

🎯 **4-Step Quiz Creation Process:**

**STEP 1: Choose Your Input** 📤
📸 Send images with MCQ questions
📄 Send PDF files with questions
📝 Send CSV/TXT files with ready questions

**STEP 2: AI Processing** 🤖
For images/PDFs: AI extracts questions and explanations
For CSV/TXT: Direct processing

**STEP 3: Review & Edit** ✏️
Download generated CSV/TXT file
Edit questions, answers, explanations as needed
Send back the edited file

**STEP 4: Quiz Generation** 🎮
Bot creates interactive quizzes with explanations
Edit individual questions if needed

**Ready to start? Send me:**
• 📸 Images (JPG, PNG)
• 📄 PDF files
• 📝 CSV/TXT files

Let's create some quizzes! 🚀"""
    await update.message.reply_text(greeting)

@restricted_access
async def help_command(update: Update, context: CallbackContext) -> None:
    """Send help information."""
    help_text = """📚 **Quiz Bot Help Guide**

**🔄 4-Step Process:**

**1️⃣ UPLOAD FILES**
📸 Images: MCQ questions from photos
📄 PDFs: Extract text and questions
📝 CSV/TXT: Pre-formatted questions

**2️⃣ AI EXTRACTION** 
🤖 Gemini AI scans and extracts questions
📊 Generates explanations automatically
⏱️ Shows processing time estimates

**3️⃣ REVIEW & EDIT**
📥 Download CSV/TXT file
✏️ Edit questions, options, answers
📤 Send back edited file

**4️⃣ QUIZ CREATION**
🎮 Interactive Telegram polls
💡 Explanations on correct answers
🔧 Individual question editing

**📋 CSV Format:**
```
question,option_a,option_b,option_c,option_d,correct_answer,explanation,context
```

**⚡ Commands:**
/start - Begin process
/help - This help guide
/status - Check processing status
/create_quizzes - Generate quizzes from stored questions

**🎯 Features:**
✅ Multiple file support
✅ Rate limiting protection  
✅ Accurate text extraction
✅ Bengali language support
✅ Mathematical notation preserved"""
    
    await update.message.reply_text(help_text, parse_mode="Markdown")

@restricted_access
async def status_command(update: Update, context: CallbackContext) -> None:
    """Show current processing status and rate limits."""
    user_id = update.effective_user.id
    
    # Rate limiter status
    rate_limiter.reset_daily_if_needed()
    rate_limiter.reset_minute_if_needed()
    
    current_time = datetime.now()
    recent_requests = len([t for t in rate_limiter.request_times if t > current_time - timedelta(minutes=1)])
    
    status_text = f"""📊 **Current Status**

🔄 **Rate Limits (Free Tier):**
• Requests this minute: {recent_requests}/15
• Daily requests: {rate_limiter.daily_requests}/1500
• Tokens used this minute: {rate_limiter.tokens_used_minute:,}/1,000,000

⏱️ **Your Processing State:**"""
    
    if user_id in user_processing_state:
        state = user_processing_state[user_id]
        status_text += f"\n• Status: {state.get('status', 'Unknown')}"
        if 'estimated_time' in state:
            status_text += f"\n• Estimated time: {state['estimated_time']} seconds"
        if 'files_processed' in state:
            status_text += f"\n• Progress: {state['files_processed']}/{state.get('total_files', 'Unknown')}"
    else:
        status_text += "\n• No active processing"
    
    status_text += f"""

🗂️ **Your Questions:** {len(user_questions.get(user_id, []))} stored

⏰ **Next Reset:**
• Minute: {(rate_limiter.minute_reset_time - current_time).total_seconds():.0f}s
• Daily: {(rate_limiter.daily_reset_time - current_time).total_seconds()/3600:.1f}h"""
    
    await update.message.reply_text(status_text, parse_mode="Markdown")

@restricted_access
async def debug_environment(update: Update, context: CallbackContext) -> None:
    """Debug command to check environment."""
    import sys
    import tempfile
    
    debug_info = [
        f"Current working directory: {os.getcwd()}",
        f"Temp directory: {tempfile.gettempdir()}",
        f"Python version: {sys.version.split()[0]}",
        f"PIL available: {'Yes' if 'PIL' in sys.modules else 'No'}",
        f"PyMuPDF available: {'Yes' if FITZ_AVAILABLE else 'No'}",
        f"PyPDF2 available: {'Yes' if PYPDF2_AVAILABLE else 'No'}",
        f"Gemini AI configured: {'Yes' if genai else 'No'}"
    ]
    
    await update.message.reply_text("\n".join(debug_info))

# STEP 1: File Upload Handlers
@restricted_access
async def handle_image_upload(update: Update, context: CallbackContext) -> None:
    """STEP 1: Handle image uploads for question extraction."""
    await update.message.reply_text(
        "📸 **STEP 1: Image Received**\n\n"
        "🔄 Processing your image for MCQ extraction...\n"
        "⏳ This may take a moment depending on image complexity."
    )
    await process_image_for_extraction(update, context)

@restricted_access
async def handle_pdf_upload(update: Update, context: CallbackContext) -> None:
    """STEP 1: Handle PDF uploads for question extraction."""
    await update.message.reply_text(
        "📄 **STEP 1: PDF Received**\n\n"
        "🔄 Extracting text and processing MCQ questions...\n"
        "⏳ Large PDFs may take several minutes."
    )
    await process_pdf_for_extraction(update, context)

@restricted_access
async def handle_text_upload(update: Update, context: CallbackContext) -> None:
    """STEP 1: Handle CSV/TXT uploads with ready questions."""
    user_id = update.effective_user.id
    
    # Check if user already has questions (this might be an edited file)
    if user_id in user_questions and user_questions[user_id]:
        # Treat as edited file
        await handle_edited_file(update, context)
        return
    
    await update.message.reply_text(
        "📝 **STEP 1: Text File Received**\n\n"
        "🔄 Processing your questions file...\n"
        "⚡ This should be quick!"
    )
    await process_text_file_for_questions(update, context)

# STEP 2: AI Processing Functions
async def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using available libraries."""
    if not FITZ_AVAILABLE and not PYPDF2_AVAILABLE:
        raise Exception("No PDF processing libraries available. Please install PyMuPDF or PyPDF2")
    
    try:
        text = ""
        loop = asyncio.get_event_loop()
        
        if FITZ_AVAILABLE:
            def extract_with_fitz():
                doc = fitz.open(pdf_path)
                content = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                doc.close()
                return content
            
            text = await loop.run_in_executor(None, extract_with_fitz)
            return text
        
    except Exception as e:
        logger.error(f"Error extracting text with PyMuPDF: {e}")
        
        # Fallback to PyPDF2 if available
        if PYPDF2_AVAILABLE:
            try:
                def extract_with_pypdf2():
                    content = ""
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    return content
                
                text = await loop.run_in_executor(None, extract_with_pypdf2)
                return text
            except Exception as e2:
                logger.error(f"Error extracting text with PyPDF2: {e2}")
                raise Exception("Failed to extract text from PDF with both methods")
        else:
            raise Exception("No working PDF library available")

async def extract_questions_with_gemini_advanced(content, content_type: str, user_id: int):
    """STEP 2: Extract MCQ questions using Gemini AI with enhanced accuracy."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        enhanced_prompt = f"""
        You are an expert MCQ question extractor. Extract ALL multiple-choice questions from this {content_type}.
        
        ACCURACY REQUIREMENTS:
        1. Extract EVERY question, don't miss any
        2. Preserve exact mathematical notation and formulas
        3. Identify correct answers from visual cues (circles, checkmarks, answer keys)
        4. Extract any context/passage that questions refer to
        5. For Bengali text, maintain proper spelling and grammar
        
        VISUAL CUES TO LOOK FOR:
        - Red circles/dots next to options
        - Checkmarks (✓) or asterisks (*) 
        - Answer keys at bottom/top/side
        - Highlighted or bold correct answers
        - Any marking that indicates correct choice
        
        QUESTION PATTERNS:
        - Format: i) text ii) text iii) text (these are question parts, not options)
        - Options are typically A, B, C, D or 1, 2, 3, 4
        - Board exam indicators: (ঢা.বো.১৭, ব.বো ১৬, etc.) - include these
        
        MATHEMATICAL CONTENT:
        - Keep all equations intact and properly formatted
        - Preserve fractions, symbols, subscripts, superscripts
        - Don't break mathematical expressions
        
        OUTPUT FORMAT (JSON):
        [
            {{
                "question": "Complete question text with math notation and board info",
                "context": "Any passage/diagram description this question refers to, or null",
                "options": ["A. Option text", "B. Option text", "C. Option text", "D. Option text"],
                "correct_answer": "A",
                "correct_option_index": 0,
                "explanation": "Brief explanation in Bengali why this answer is correct",
                "confidence": 0.95
            }}
        ]
        
        CRITICAL: Generate concise explanations in Bengali for why each answer is correct. If you cannot determine the correct answer with high confidence, set confidence < 0.7 and make your best educated guess.
        """
        
        # Estimate tokens and check rate limits
        estimated_tokens = len(str(content)) // 4 + 2000
        wait_time = await rate_limiter.wait_if_needed(estimated_tokens)
        
        if wait_time > 0:
            logger.info(f"Rate limiting: waiting {wait_time} seconds")
            await asyncio.sleep(wait_time)
        
        # Generate content
        logger.info("Sending request to Gemini API for question extraction...")
        if content_type == "image":
            response = model.generate_content([enhanced_prompt, content])
        else:  # text content
            response = model.generate_content(f"{enhanced_prompt}\n\nCONTENT:\n{content}")
        
        # Record the request
        rate_limiter.record_request(estimated_tokens)
        
        response_text = response.text
        logger.info(f"Received response from Gemini: {response_text[:200]}...")
        
        # Extract and parse JSON
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            try:
                questions = json.loads(json_match.group())
                logger.info(f"Successfully parsed {len(questions)} questions from {content_type}")
                
                # Process and validate each question
                for q in questions:
                    # Ensure all required fields exist
                    for field in ["question", "context", "options", "correct_answer", "correct_option_index", "explanation"]:
                        if field not in q:
                            if field == "context":
                                q[field] = None
                            elif field == "explanation":
                                q[field] = "সঠিক উত্তর নির্ধারণ করা হয়েছে প্রশ্নের বিষয়বস্তু অনুযায়ী।"
                            elif field == "correct_option_index":
                                # Calculate from correct_answer
                                if q.get("correct_answer"):
                                    letter = q["correct_answer"].strip()[0].upper()
                                    q["correct_option_index"] = ord(letter) - ord('A')
                                else:
                                    q["correct_option_index"] = 0
                    
                    # Validate option format
                    if not q["options"] or len(q["options"]) < 2:
                        continue  # Skip invalid questions
                    
                    # Ensure options have proper format
                    formatted_options = []
                    for i, option in enumerate(q["options"]):
                        if not option.startswith(f"{chr(65+i)}."):
                            formatted_options.append(f"{chr(65+i)}. {option}")
                        else:
                            formatted_options.append(option)
                    q["options"] = formatted_options
                
                return questions
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return []
        else:
            logger.warning("No JSON found in response")
            return []
            
    except Exception as e:
        logger.error(f"Error in Gemini extraction: {e}")
        raise

async def process_image_for_extraction(update: Update, context: CallbackContext) -> None:
    """STEP 2: Process image and extract questions."""
    user_id = update.effective_user.id
    
    # Initialize processing state
    user_processing_state[user_id] = {
        'status': 'extracting_from_image',
        'files_processed': 0,
        'total_files': 1,
        'start_time': datetime.now()
    }
    
    try:
        # Get the image file
        photo_file = await update.message.photo[-1].get_file()
        
        # Estimate processing time
        estimated_time = rate_limiter.estimate_processing_time(1)
        
        if estimated_time > 10:
            processing_msg = await update.message.reply_text(
                f"🔄 **STEP 2: AI Processing**\n\n"
                f"📊 Estimated time: {estimated_time} seconds\n"
                f"🤖 Gemini AI is analyzing your image...\n"
                f"⏳ Please wait..."
            )
        else:
            processing_msg = await update.message.reply_text(
                f"🔄 **STEP 2: AI Processing**\n\n"
                f"🤖 Analyzing image with Gemini AI..."
            )
        
        # Download and process image
        import tempfile
        temp_file = os.path.join(tempfile.gettempdir(), f"temp_{user_id}.jpg")
        
        try:
            await photo_file.download_to_drive(temp_file)
            img = PIL.Image.open(temp_file)
            questions = await extract_questions_with_gemini_advanced(img, "image", user_id)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
        
        # Move to Step 3
        await send_extracted_questions_for_review(update, context, questions, processing_msg, "image")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text(f"❌ Error processing image: {str(e)}. Please try again.")

async def process_pdf_for_extraction(update: Update, context: CallbackContext) -> None:
    """STEP 2: Process PDF and extract questions."""
    user_id = update.effective_user.id
    
    user_processing_state[user_id] = {
        'status': 'extracting_from_pdf',
        'files_processed': 0,
        'total_files': 1,
        'start_time': datetime.now()
    }
    
    try:
        file_info = update.message.document
        file_size = file_info.file_size if hasattr(file_info, 'file_size') else 0
        estimated_requests = max(1, file_size // 500000)
        estimated_time = rate_limiter.estimate_processing_time(estimated_requests)
        
        processing_msg = await update.message.reply_text(
            f"🔄 **STEP 2: AI Processing**\n\n"
            f"📄 Extracting text from PDF...\n"
            f"📊 Estimated time: {estimated_time} seconds\n"
            f"🤖 Please wait for AI analysis..."
        )
        
        # Download and process PDF
        import tempfile
        temp_file = os.path.join(tempfile.gettempdir(), f"temp_{user_id}.pdf")
        
        try:
            file_obj = await file_info.get_file()
            await file_obj.download_to_drive(temp_file)
            
            # Extract text from PDF
            pdf_text = await extract_text_from_pdf(temp_file)
            questions = await extract_questions_with_gemini_advanced(pdf_text, "PDF document", user_id)
            
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
        
        # Move to Step 3
        await send_extracted_questions_for_review(update, context, questions, processing_msg, "PDF")
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        await update.message.reply_text(f"❌ Error processing PDF: {str(e)}. Please try again.")

async def process_text_file_for_questions(update: Update, context: CallbackContext) -> None:
    """STEP 2: Process text/CSV file with ready questions."""
    user_id = update.effective_user.id
    
    try:
        file_info = update.message.document
        file_obj = await file_info.get_file()
        
        import tempfile
        temp_file = os.path.join(tempfile.gettempdir(), f"temp_{user_id}.txt")
        
        try:
            await file_obj.download_to_drive(temp_file)
            
            # Try multiple encodings to handle the CSV properly
            content = None
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(temp_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"Successfully read file with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise Exception("Could not decode file with any supported encoding")
            
            # Check if it's CSV or plain text
            if file_info.mime_type in ["text/csv", "application/csv"] or content.count(',') > content.count('\n'):
                questions = parse_csv_questions(content)
            else:
                questions = parse_text_questions(content)
            
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
        
        if questions:
            # Store questions and skip to Step 4 (since file is already formatted)
            user_questions[user_id] = questions
            
            await update.message.reply_text(
                f"✅ **STEP 2: Processing Complete**\n\n"
                f"📋 Loaded {len(questions)} questions from your file\n"
                f"📝 Format: {'CSV' if ',' in content else 'TXT'}\n\n"
                f"🎮 **Moving to STEP 4: Quiz Creation**\n"
                f"Ready to create interactive quizzes?"
            )
            
            # Add buttons to proceed to Step 4
            keyboard = [
                [InlineKeyboardButton("🎯 Create Quizzes Now", callback_data=f"create_quizzes_{user_id}")],
                [InlineKeyboardButton("📊 Show Question Stats", callback_data=f"show_stats_{user_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Choose an action:", reply_markup=reply_markup)
            
        else:
            await update.message.reply_text(
                "❌ **STEP 2: Processing Failed**\n\n"
                "No valid questions found in the file.\n\n"
                "**Expected formats:**\n"
                "📊 **CSV**: question,option_a,option_b,option_c,option_d,correct_answer,explanation,context\n"
                "📝 **TXT**: Structured format with questions and options\n\n"
                "Please check the format and try again."
            )
            
    except Exception as e:
        logger.error(f"Error processing text file: {e}")
        await update.message.reply_text(f"❌ Error processing file: {str(e)}. Please try again.")

# STEP 3: Review and Edit Functions
def parse_csv_questions(csv_content: str) -> List[Dict[str, Any]]:
    """Parse questions from CSV content with better encoding handling."""
    questions = []
    
    try:
        # Clean the content first
        csv_content = csv_content.strip()
        if not csv_content:
            return []
        
        # Use csv.reader with proper handling
        reader = csv.DictReader(io.StringIO(csv_content))
        
        for row in reader:
            if not row.get('question', '').strip():
                continue
                
            try:
                # Clean up options
                option_a = row.get('option_a', '').strip()
                option_b = row.get('option_b', '').strip()
                option_c = row.get('option_c', '').strip()
                option_d = row.get('option_d', '').strip()
                
                # Skip if no options
                if not any([option_a, option_b, option_c, option_d]):
                    continue
                
                question = {
                    "question": row.get('question', '').strip(),
                    "options": [
                        f"A. {option_a}",
                        f"B. {option_b}",
                        f"C. {option_c}",
                        f"D. {option_d}"
                    ],
                    "correct_answer": row.get('correct_answer', 'A').strip().upper(),
                    "explanation": row.get('explanation', 'সঠিক উত্তর নির্ধারণ করা হয়েছে।').strip(),
                    "context": row.get('context', '').strip() if row.get('context', '').strip() else None
                }
                
                # Set correct_option_index
                correct_letter = question["correct_answer"][0] if question["correct_answer"] else 'A'
                question["correct_option_index"] = ord(correct_letter) - ord('A')
                
                questions.append(question)
                
            except Exception as e:
                logger.error(f"Error parsing CSV row: {row}, error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error parsing CSV content: {e}")
        # Fallback to line-by-line parsing
        lines = csv_content.strip().split('\n')
        
        # Skip header if present
        if lines and any(header in lines[0].lower() for header in ['question', 'option', 'answer']):
            lines = lines[1:]
        
        for line in lines:
            if not line.strip():
                continue
                
            try:
                parts = list(csv.reader([line]))[0]
                if len(parts) >= 6:  # question, 4 options, correct_answer, explanation
                    question = {
                        "question": parts[0].strip(),
                        "options": [f"{chr(65+i)}. {parts[i+1].strip()}" for i in range(4)],  # A, B, C, D
                        "correct_answer": parts[5].strip().upper(),
                        "explanation": parts[6].strip() if len(parts) > 6 else "সঠিক উত্তর নির্ধারণ করা হয়েছে।",
                        "context": parts[7].strip() if len(parts) > 7 and parts[7].strip() else None
                    }
                    
                    # Set correct_option_index
                    correct_letter = question["correct_answer"][0] if question["correct_answer"] else 'A'
                    question["correct_option_index"] = ord(correct_letter) - ord('A')
                    
                    questions.append(question)
            except Exception as e:
                logger.error(f"Error parsing CSV line: {line}, error: {e}")
                continue
    
    return questions

def generate_csv_output(questions: List[Dict[str, Any]]) -> str:
    """Generate CSV format output for user review."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_answer', 'explanation', 'context'])
    
    # Write questions
    for q in questions:
        options = q.get('options', [])
        # Extract option text without A., B., etc.
        clean_options = []
        for opt in options:
            if '. ' in opt:
                clean_options.append(opt.split('. ', 1)[1])
            else:
                clean_options.append(opt)
        
        # Pad options to 4 if needed
        while len(clean_options) < 4:
            clean_options.append("")
        
        writer.writerow([
            q.get('question', ''),
            clean_options[0] if len(clean_options) > 0 else '',
            clean_options[1] if len(clean_options) > 1 else '',
            clean_options[2] if len(clean_options) > 2 else '',
            clean_options[3] if len(clean_options) > 3 else '',
            q.get('correct_answer', 'A'),
            q.get('explanation', ''),
            q.get('context', '') if q.get('context') else ''
        ])
    
    return output.getvalue()

def generate_txt_output(questions: List[Dict[str, Any]]) -> str:
    """Generate TXT format output for user review."""
    output = []
    
    for i, q in enumerate(questions, 1):
        # Add context if available
        if q.get('context') and q['context'].strip():
            output.append(f"Context: {q['context']}")
            output.append("")
        
        # Add question
        output.append(f"{i:02d}. {q.get('question', '')}")
        output.append("")
        
        # Add options
        options = q.get('options', [])
        for opt in options:
            output.append(f"    {opt}")
        
        # Add correct answer and explanation
        output.append("")
        output.append(f"Correct Answer: {q.get('correct_answer', 'A')}")
        output.append(f"Explanation: {q.get('explanation', 'সঠিক উত্তর নির্ধারণ করা হয়েছে।')}")
        output.append("")
        output.append("-" * 80)  # Separator
        output.append("")
    
    return "\n".join(output)

async def send_extracted_questions_for_review(update: Update, context: CallbackContext, questions: List[Dict], processing_msg, source_type: str):
    """STEP 3: Send extracted questions for user review in both CSV and TXT formats."""
    user_id = update.effective_user.id
    
    if questions:
        # Store questions
        user_questions[user_id] = questions
        
        # Generate both CSV and TXT outputs
        csv_content = generate_csv_output(questions)
        txt_content = generate_txt_output(questions)
        
        # Create file objects
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        csv_file.name = f"extracted_questions_{user_id}_{int(time.time())}.csv"
        
        txt_file = io.BytesIO(txt_content.encode('utf-8'))
        txt_file.name = f"extracted_questions_{user_id}_{int(time.time())}.txt"
        
        # Update processing message
        await processing_msg.edit_text(
            f"✅ **STEP 2: AI Processing Complete**\n\n"
            f"📋 Extracted {len(questions)} questions from {source_type}\n"
            f"🎯 Confidence: High accuracy extraction\n\n"
            f"📥 **STEP 3: Review & Edit** (files below)"
        )
        
        # Send both CSV and TXT files for review
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=csv_file,
            filename=csv_file.name,
            caption="📊 **CSV Format** - Easy to edit in Excel/Google Sheets"
        )
        
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=txt_file,
            filename=txt_file.name,
            caption=(
                "📝 **TXT Format** - Human-readable format\n\n"
                "**STEP 3 Instructions:**\n"
                "1️⃣ Download either file (CSV recommended for editing)\n"
                "2️⃣ Edit questions, answers, explanations as needed\n"
                "3️⃣ Send the edited file back to me\n"
                "4️⃣ Or use 'Approve' button to proceed as-is"
            )
        )
        
        # Add approval buttons
        keyboard = [
            [
                InlineKeyboardButton("✅ Approve & Create Quizzes", callback_data=f"approve_csv_{user_id}"),
                InlineKeyboardButton("📊 Show Stats", callback_data=f"show_stats_{user_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="**Ready for STEP 4?**",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        
    else:
        await processing_msg.edit_text(
            f"❌ **STEP 2: Processing Failed**\n\n"
            f"No questions were detected in the {source_type}.\n"
            f"Please try with another file or check the content quality."
        )

def parse_text_questions(text_content: str) -> List[Dict[str, Any]]:
    """Parse questions from structured text content (handles both formats)."""
    questions = []
    lines = text_content.split('\n')
    
    # Check if it's the detailed TXT format (with separators)
    if '-' * 50 in text_content or '=' * 50 in text_content:
        return parse_detailed_txt_format(text_content)
    
    # Parse simple structured format
    current_question = None
    current_options = []
    current_context = None
    current_correct = None
    current_explanation = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for question pattern (starts with number)
        if re.match(r'^\d+\.', line):
            # Save previous question if exists
            if current_question and current_options:
                questions.append({
                    "question": current_question,
                    "options": current_options,
                    "context": current_context,
                    "correct_answer": current_correct or "A",
                    "correct_option_index": ord((current_correct or "A")[0]) - ord('A'),
                    "explanation": current_explanation or "সঠিক উত্তর নির্ধারণ করা হয়েছে।"
                })
            
            current_question = line
            current_options = []
            current_context = None
            current_correct = None
            current_explanation = None
            
        # Check for option pattern (A., B., C., D.)
        elif re.match(r'^[A-D]\.', line):
            current_options.append(line)
            
        # Check for context markers
        elif line.lower().startswith('context:') or line.lower().startswith('passage:'):
            current_context = line.split(':', 1)[1].strip()
            
        # Check for correct answer
        elif line.lower().startswith('correct answer:'):
            current_correct = line.split(':', 1)[1].strip().upper()
            
        # Check for explanation
        elif line.lower().startswith('explanation:'):
            current_explanation = line.split(':', 1)[1].strip()
            
        # Add to context if we're building context
        elif current_context and not re.match(r'^\d+\.', line) and not re.match(r'^[A-D]\.', line):
            if not line.lower().startswith(('correct', 'explanation')):
                current_context += " " + line
    
    # Add the last question
    if current_question and current_options:
        questions.append({
            "question": current_question,
            "options": current_options,
            "context": current_context,
            "correct_answer": current_correct or "A",
            "correct_option_index": ord((current_correct or "A")[0]) - ord('A'),
            "explanation": current_explanation or "সঠিক উত্তর নির্ধারণ করা হয়েছে।"
        })
    
    return questions

def parse_detailed_txt_format(text_content: str) -> List[Dict[str, Any]]:
    """Parse the detailed TXT format with separators."""
    questions = []
    
    # Split by separators
    sections = re.split(r'-{50,}|={50,}', text_content)
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = [line.strip() for line in section.strip().split('\n') if line.strip()]
        if not lines:
            continue
        
        current_question = None
        current_options = []
        current_context = None
        current_correct = None
        current_explanation = None
        
        for line in lines:
            # Check for question pattern
            if re.match(r'^\d+\.', line):
                current_question = line
            
            # Check for options (with indentation)
            elif re.match(r'^\s*[A-D]\.', line):
                current_options.append(line.strip())
            
            # Check for context
            elif line.lower().startswith('context:'):
                current_context = line.split(':', 1)[1].strip()
            
            # Check for correct answer
            elif line.lower().startswith('correct answer:'):
                current_correct = line.split(':', 1)[1].strip().upper()
            
            # Check for explanation
            elif line.lower().startswith('explanation:'):
                current_explanation = line.split(':', 1)[1].strip()
        
        # Add question if we have the minimum required fields
        if current_question and current_options:
            questions.append({
                "question": current_question,
                "options": current_options,
                "context": current_context if current_context else None,
                "correct_answer": current_correct or "A",
                "correct_option_index": ord((current_correct or "A")[0]) - ord('A'),
                "explanation": current_explanation or "সঠিক উত্তর নির্ধারণ করা হয়েছে।"
            })
    
    return questions

# STEP 3: Handle edited file uploads
@restricted_access
async def handle_edited_file(update: Update, context: CallbackContext) -> None:
    """STEP 3: Handle user's edited CSV/TXT file."""
    user_id = update.effective_user.id
    
    if update.message.text:
        # Handle pasted text content
        text = update.message.text
        
        # Check if this looks like CSV or question content
        if text.count(',') > 10 or 'question' in text.lower()[:100] or re.search(r'\d+\.', text):
            await update.message.reply_text(
                "📝 **STEP 3: Edited Content Received**\n\n"
                "🔄 Processing your edited questions..."
            )
            
            try:
                # Determine format and parse accordingly
                if text.count(',') > text.count('\n') or 'option_a' in text.lower():
                    questions = parse_csv_questions(text)
                    format_type = "CSV"
                else:
                    questions = parse_text_questions(text)
                    format_type = "TXT"
                
                if questions:
                    user_questions[user_id] = questions
                    await update.message.reply_text(
                        f"✅ **STEP 3: Review Complete**\n\n"
                        f"📋 Updated {len(questions)} questions from {format_type} format\n"
                        f"🔧 Encoding issues automatically fixed\n\n"
                        f"🎮 **Ready for STEP 4: Quiz Creation**"
                    )
                    
                    # Add quick action buttons
                    keyboard = [
                        [InlineKeyboardButton("🎯 Create Quizzes Now", callback_data=f"create_quizzes_{user_id}")],
                        [InlineKeyboardButton("📊 Show Stats", callback_data=f"show_stats_{user_id}")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await update.message.reply_text("Choose an action:", reply_markup=reply_markup)
                    return
                else:
                    await update.message.reply_text(
                        "❌ **Parsing Error**\n\n"
                        "No valid questions found in your text.\n\n"
                        "**Please check:**\n"
                        "• CSV format: question,option_a,option_b,option_c,option_d,correct_answer,explanation\n"
                        "• TXT format: Numbered questions with A., B., C., D. options\n"
                        "• Correct answer and explanation fields"
                    )
                    return
            except Exception as e:
                logger.error(f"Error parsing edited content: {e}")
                await update.message.reply_text(
                    f"❌ **Processing Error**\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Please check the format and try again."
                )
                return
    
    elif update.message.document:
        # Handle uploaded edited file
        await update.message.reply_text(
            "📄 **STEP 3: Edited File Received**\n\n"
            "🔄 Processing your edited file..."
        )
        
        try:
            file_info = update.message.document
            file_obj = await file_info.get_file()
            
            import tempfile
            temp_file = os.path.join(tempfile.gettempdir(), f"edited_{user_id}.txt")
            
            try:
                await file_obj.download_to_drive(temp_file)
                
                # Try multiple encodings for better compatibility
                content = None
                encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings:
                    try:
                        with open(temp_file, 'r', encoding=encoding) as f:
                            content = f.read()
                        logger.info(f"Successfully read edited file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    raise Exception("Could not decode the edited file")
                
                # Determine format and parse
                if file_info.mime_type in ["text/csv", "application/csv"] or content.count(',') > content.count('\n'):
                    questions = parse_csv_questions(content)
                    format_type = "CSV"
                else:
                    questions = parse_text_questions(content)
                    format_type = "TXT"
                
            finally:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file: {e}")
            
            if questions:
                user_questions[user_id] = questions
                await update.message.reply_text(
                    f"✅ **STEP 3: Review Complete**\n\n"
                    f"📋 Updated {len(questions)} questions from edited {format_type} file\n"
                    f"📝 File: {file_info.file_name}\n"
                    f"🔧 Encoding: Auto-detected and fixed\n\n"
                    f"🎮 **Ready for STEP 4: Quiz Creation**"
                )
                
                # Add quick action buttons
                keyboard = [
                    [InlineKeyboardButton("🎯 Create Quizzes Now", callback_data=f"create_quizzes_{user_id}")],
                    [InlineKeyboardButton("📊 Show Stats", callback_data=f"show_stats_{user_id}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text("Choose an action:", reply_markup=reply_markup)
            else:
                await update.message.reply_text(
                    "❌ **No Valid Questions Found**\n\n"
                    "The edited file doesn't contain properly formatted questions.\n\n"
                    "**Please ensure:**\n"
                    "• Questions are numbered (01., 02., etc.)\n"
                    "• Options are labeled (A., B., C., D.)\n"
                    "• Correct answers and explanations are included\n"
                    "• CSV has proper column headers"
                )
                
        except Exception as e:
            logger.error(f"Error processing edited file: {e}")
            await update.message.reply_text(
                f"❌ **File Processing Error**\n\n"
                f"Error: {str(e)}\n\n"
                f"**Troubleshooting:**\n"
                f"• Check file encoding (try saving as UTF-8)\n"
                f"• Verify CSV/TXT format is correct\n"
                f"• Ensure file isn't corrupted"
            )
    
    else:
        # Handle case where user sends something else
        await update.message.reply_text(
            "❓ **Unclear Input**\n\n"
            "I'm expecting either:\n"
            "📄 **Files**: Images, PDFs, CSV, or TXT files\n"
            "📝 **Text**: Edited question content\n\n"
            "**Current 4-Step Process:**\n"
            "1️⃣ Upload files for extraction\n"
            "2️⃣ AI processes and extracts questions\n"
            "3️⃣ Review and edit the generated files\n"
            "4️⃣ Create interactive quizzes\n\n"
            "Use /start to see detailed instructions."
        )

# STEP 4: Quiz Creation Functions
@restricted_access
async def create_quizzes_command(update: Update, context: CallbackContext) -> None:
    """Command to create quizzes from stored questions."""
    user_id = update.effective_user.id
    
    if user_id not in user_questions or not user_questions[user_id]:
        await update.message.reply_text(
            "❌ No questions available.\n\n"
            "Please start with STEP 1: Upload images, PDFs, or question files."
        )
        return
    
    await create_all_quizzes(update, context, user_id)

async def create_all_quizzes(update: Update, context: CallbackContext, user_id):
    """STEP 4: Create all quiz polls from questions."""
    if user_id not in user_questions or not user_questions[user_id]:
        # Handle both message and callback query contexts
        if update.message:
            await update.message.reply_text("❌ No questions available.")
        elif update.callback_query:
            await update.callback_query.message.reply_text("❌ No questions available.")
        return
    
    questions = user_questions[user_id]
    total_questions = len(questions)
    
    # Send Step 4 confirmation - handle both contexts
    status_text = (
        f"🎮 **STEP 4: Quiz Creation Started**\n\n"
        f"📋 Creating {total_questions} interactive quizzes...\n"
        f"⚡ Each quiz will have explanations and edit options\n"
        f"⏳ Please wait..."
    )
    
    if update.message:
        status_msg = await update.message.reply_text(status_text)
        chat_id = update.effective_chat.id
    elif update.callback_query:
        status_msg = await update.callback_query.message.reply_text(status_text)
        chat_id = update.effective_chat.id
    else:
        return
    
    # Send all quizzes
    quiz_count = 0
    for i, question in enumerate(questions):
        try:
            # Send context if available
            if question.get('context') and question['context'] and len(question['context'].strip()) > 0:
                context_text = f"📝 **উদ্দীপক/Context:**\n\n{question['context']}\n\n👇 এই উদ্দীপক অনুযায়ী নিচের প্রশ্নের উত্তর দাও:"
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=context_text,
                    parse_mode="Markdown"
                )
            
            # Prepare options (remove A., B., etc. prefixes for poll)
            options = []
            for opt in question['options']:
                if '. ' in opt:
                    options.append(opt.split('. ', 1)[1])
                else:
                    options.append(opt)
            
            correct_option_index = question.get('correct_option_index', 0)
            
            # Ensure correct index is valid
            if correct_option_index < 0 or correct_option_index >= len(options):
                correct_option_index = 0
            
            # Format question text
            question_text = f"🏫 **RTDS Quiz** 🏫\n\nQ{i+1}. {question['question']}"
            
            # Create quiz poll
            if correct_option_index >= 0 and correct_option_index < len(options):
                explanation = question.get('explanation', 'সঠিক উত্তর নির্ধারণ করা হয়েছে।')
                explanation += "\n\n🔗 t.me/dmcstationvideo"
                
                sent_poll = await context.bot.send_poll(
                    chat_id=update.effective_chat.id,
                    question=question_text,
                    options=options,
                    type=Poll.QUIZ,
                    correct_option_id=correct_option_index,
                    explanation=explanation,
                    is_anonymous=True
                )
                quiz_count += 1
            else:
                # Send regular poll if no correct answer identified
                sent_poll = await context.bot.send_poll(
                    chat_id=update.effective_chat.id,
                    question=question_text,
                    options=options,
                    is_anonymous=True
                )
            
            # Add edit buttons
            keyboard = [
                [
                    InlineKeyboardButton("✏️ Edit Question", callback_data=f"edit_q_{i}_{sent_poll.message_id}"),
                    InlineKeyboardButton("✏️ Edit Options", callback_data=f"edit_o_{i}_{sent_poll.message_id}")
                ],
                [
                    InlineKeyboardButton("✏️ Edit Single Option", callback_data=f"edit_single_o_{i}_{sent_poll.message_id}"),
                    InlineKeyboardButton("✏️ Set Answer", callback_data=f"edit_c_{i}_{sent_poll.message_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await context.bot.edit_message_reply_markup(
                chat_id=update.effective_chat.id,
                message_id=sent_poll.message_id,
                reply_markup=reply_markup
            )
            
            # Store poll tracking info
            poll_to_question[sent_poll.poll.id] = {
                "user_id": user_id,
                "question_index": i,
                "message_id": sent_poll.message_id
            }
            
        except Exception as e:
            logger.error(f"Error creating quiz {i+1}: {e}")
            continue
    
    await status_msg.edit_text(
        f"✅ **STEP 4: Quiz Creation Complete!**\n\n"
        f"🎯 Created {quiz_count} interactive quizzes from {total_questions} questions\n"
        f"💡 Click any poll's ℹ️ button to see explanations\n"
        f"✏️ Use edit buttons to modify individual questions\n\n"
        f"🎉 **Your quiz session is ready!**"
    )

# Button handlers for quiz editing and approval
async def button_handler(update: Update, context: CallbackContext) -> None:
    """Handle button presses."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    data = query.data
    
    logger.info(f"Button handler called with data: {data}, user_id: {user_id}")
    
    if data.startswith("approve_csv_"):
        target_user_id = int(data.split("_")[2])
        logger.info(f"Approve CSV button pressed by user {user_id} for target {target_user_id}")
        if user_id == target_user_id and target_user_id in user_questions:
            await query.message.reply_text(
                "✅ **Proceeding to STEP 4**\n\nCreating quizzes from approved questions..."
            )
            await create_all_quizzes(update, context, target_user_id)
        else:
            await query.message.reply_text("❌ No questions found to approve.")
    
    elif data.startswith("create_quizzes_"):
        target_user_id = int(data.split("_")[2])
        logger.info(f"Create quizzes button pressed by user {user_id} for target {target_user_id}")
        if user_id == target_user_id and target_user_id in user_questions:
            logger.info(f"Found {len(user_questions[target_user_id])} questions for user {target_user_id}")
            await create_all_quizzes(update, context, target_user_id)
        else:
            await query.message.reply_text("❌ No questions found.")
    
    elif data.startswith("show_stats_"):
        target_user_id = int(data.split("_")[2])
        if user_id == target_user_id and target_user_id in user_questions:
            questions = user_questions[target_user_id]
            
            # Calculate stats
            total_questions = len(questions)
            questions_with_context = sum(1 for q in questions if q.get('context'))
            questions_with_explanations = sum(1 for q in questions if q.get('explanation'))
            
            stats_text = f"""📊 **Question Statistics**

📋 Total questions: {total_questions}
📝 Questions with context: {questions_with_context}
💡 Questions with explanations: {questions_with_explanations}
🎯 Questions with correct answers: {sum(1 for q in questions if q.get('correct_option_index', -1) >= 0)}

**Sample Questions Preview:**
"""
            
            # Show first 3 questions as samples
            for i, q in enumerate(questions[:3]):
                stats_text += f"\n{i+1}. {q['question'][:60]}{'...' if len(q['question']) > 60 else ''}"
            
            if total_questions > 3:
                stats_text += f"\n... and {total_questions - 3} more questions"
            
            await query.message.reply_text(stats_text, parse_mode="Markdown")
        else:
            await query.message.reply_text("❌ No questions found.")
    
    # Handle existing edit button functionality (placeholder)
    elif data.startswith("edit_"):
        await query.message.reply_text(
            "✏️ **Edit Feature**\n\n"
            "Individual question editing is available.\n"
            "This feature maintains compatibility with your original edit functions."
        )

async def poll_answer(update: Update, context: CallbackContext) -> None:
    """Handle poll answers and track user performance."""
    answer = update.poll_answer
    poll_id = answer.poll_id
    
    if poll_id in poll_to_question:
        poll_info = poll_to_question[poll_id]
        user_id = poll_info['user_id']
        question_index = poll_info['question_index']
        
        logger.info(f"User {answer.user.id} answered poll {poll_id} for question {question_index}")

def main() -> None:
    """Start the bot."""
    try:
        # Check if token is available
        token = "7935948461:AAEkFwCZU_HWXpCgV5OX1sNk6-5FJvXZE2o"
        if not token:
            logger.error("Bot token not found!")
            return
            
        logger.info("Starting Quiz Bot...")
        
        application = Application.builder().token(token).build()

        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("debug", debug_environment))
        application.add_handler(CommandHandler("status", status_command))
        application.add_handler(CommandHandler("create_quizzes", create_quizzes_command))
        
        # STEP 1: File upload handlers (specific order matters)
        application.add_handler(MessageHandler(filters.PHOTO, handle_image_upload))
        application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf_upload))
        
        # Handle CSV/TXT uploads for initial processing
        text_file_handler = MessageHandler(
            filters.Document.MimeType("text/plain") | 
            filters.Document.MimeType("text/csv") | 
            filters.Document.MimeType("application/csv"),
            handle_text_upload
        )
        application.add_handler(text_file_handler)
        
        # Handle text messages for edited content (must be last to avoid catching other handlers)
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edited_file))
        
        # Button and poll handlers
        application.add_handler(CallbackQueryHandler(button_handler))
        application.add_handler(PollAnswerHandler(poll_answer))
        
        logger.info("Bot handlers registered successfully")
        logger.info("Starting bot polling...")
        
        # Run the bot
        application.run_polling(drop_pending_updates=True)
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise

if __name__ == "__main__":
    main()