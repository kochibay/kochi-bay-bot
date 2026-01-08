import os
import logging
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes, ConversationHandler
from openai import OpenAI
import base64
from io import BytesIO
from supabase import create_client, Client
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize clients
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
AI_PROVIDER = os.getenv('AI_PROVIDER', 'deepseek')

# Initialize Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except TypeError as e:
    if 'proxy' in str(e):
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        raise

# Initialize AI client
if AI_PROVIDER == 'nvidia' and NVIDIA_API_KEY:
    ai_client = OpenAI(
        api_key=NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1"
    )
    AI_MODEL = "meta/llama-3.2-90b-vision-instruct"
    logger.info("Using NVIDIA API")
elif AI_PROVIDER == 'openai' and OPENAI_API_KEY:
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    AI_MODEL = "gpt-4o-mini"
    logger.info("Using OpenAI API")
else:
    ai_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )
    AI_MODEL = "deepseek-chat"
    logger.info("Using DeepSeek API")

# Conversation states
ASK_STAFF_NAME = 0
CONFIRM_DATE = 1
EDIT_DATE = 2
EDIT_MENU = 3
EDIT_SUPPLIER = 4
EDIT_INVOICE_NUM = 5
EDIT_AMOUNT = 6
EDIT_ITEMS = 7
EDIT_ITEMS_ACTION = 8
ADD_ITEM = 9
EDIT_ITEM_FIELD = 10
SELECT_BRANCH = 11
ENTER_OTHER_BRANCH = 12
SELECT_CATEGORY = 13
ENTER_OTHER_CATEGORY = 14
CONFIRM_VAT = 15
CONFIRM_DATA = 16

# Store pending invoices and staff names
pending_invoices = {}
staff_names = {}  # user_id: staff_name

# Categories and Branches
CATEGORIES = [
    "Groceries", "Wholesale", "Meat & Poultry", "Vegetables",
    "Dairy", "Dry Goods", "Cleaning Supplies", "Beverages",
    "Frozen Foods", "Other"
]

BRANCHES = ["Preston", "Warrington", "Wirral", "Common for All", "Other"]

# Branch name to code mapping (matching HTML)
BRANCH_CODE_MAP = {
    "Preston": "PR",
    "Warrington": "WA",
    "Wirral": "WR",
    "Common for All": "C"
}

def get_branch_code(branch_name):
    """Convert branch name to code for database."""
    return BRANCH_CODE_MAP.get(branch_name, branch_name)  # Return as-is if not found (e.g., "Other")

def clean_supplier_name(supplier):
    """Clean up supplier name formatting."""
    if not supplier:
        return "Unknown"
    
    supplier = re.sub(r'\s+(LTD|LIMITED|PLC|COMPANY|CO\.|INC\.?)$', '', supplier, flags=re.IGNORECASE)
    supplier = supplier.title()
    
    replacements = {
        'Tesco Express': 'Tesco Express',
        'Asda': 'ASDA',
        'Aldi': 'ALDI',
        'Lidl': 'Lidl',
    }
    
    for old, new in replacements.items():
        if old.lower() in supplier.lower():
            return new
    
    return supplier.strip()

def format_currency(amount):
    """Format currency with 2 decimal places."""
    if amount is None:
        return "£0.00"
    try:
        return f"£{float(amount):.2f}"
    except:
        return "£0.00"

def parse_date_input(date_str):
    """Parse various date formats to YYYY-MM-DD."""
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Try common UK formats
    formats = [
        '%d/%m/%Y',    # 24/12/2025
        '%d-%m-%Y',    # 24-12-2025
        '%d/%m/%y',    # 24/12/25
        '%d-%m-%y',    # 24-12-25
        '%d %m %Y',    # 24 12 2025
        '%d.%m.%Y',    # 24.12.2025
        '%Y-%m-%d',    # 2025-12-24 (ISO)
    ]
    
    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            # If year is < 2000, assume 20xx
            if date_obj.year < 2000:
                date_obj = date_obj.replace(year=date_obj.year + 2000)
            return date_obj.strftime('%Y-%m-%d')
        except:
            continue
    
    return None

def format_date(date_str):
    """Format date as DD-MMM-YYYY."""
    if not date_str:
        return None
    try:
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return date_obj.strftime('%d-%b-%Y')
    except:
        try:
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime('%d-%b-%Y')
                except:
                    continue
        except:
            pass
    return date_str

def validate_date(date_str):
    """Check if date is reasonable."""
    try:
        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        today = datetime.now()
        
        if date_obj > today:
            return False, "Date is in the future"
        
        if (today - date_obj).days > 90:
            return False, f"Date is {(today - date_obj).days} days old"
        
        return True, "OK"
    except:
        return True, "OK"

def encode_image_to_base64(image_bytes):
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

async def extract_invoice_data(image_bytes):
    """Extract invoice data using configured AI API with improved prompts."""
    try:
        base64_image = encode_image_to_base64(image_bytes)
        
        current_year = datetime.now().year
        
        prompt = f"""
Analyze this invoice/receipt image and extract information in JSON format.

CRITICAL - DATE EXTRACTION (MOST IMPORTANT):
- Current year is {current_year}
- Look carefully at the date on the receipt
- Common UK format: DD/MM/YYYY (e.g., 24/12/2025)
- Look at the YEAR digits very carefully - distinguish 2024 from 2025
- The date is usually at top or bottom of receipt
- If you see a timestamp like "24/12/2025 18:42:06", the date is 24/12/2025
- Return date as YYYY-MM-DD format
- DOUBLE CHECK THE YEAR - this is critical!

ITEM EXTRACTION (CRITICAL - READ CAREFULLY):
**EACH LINE WITH A PRICE IS A SEPARATE ITEM!**

Common receipt formats you'll see:

Format 1: Each item on own line with price
```
6PC ALC O/R          12.49
MINI FLT BGR S$       1.99
```
→ This is TWO items!
- Item 1: "6PC ALC O/R", qty=1, price=12.49
- Item 2: "MINI FLT BGR S$", qty=1, price=1.99

Format 2: Description split across lines
```
RED SHALLOT ONIONS
0.9c kg @ 6.99/kg    6.57
```
→ This is ONE item (description + details on next line)

Format 3: Long description wraps
```
RAW MANGO (1.935kg @
7.99/kg)            15.46
```
→ This is ONE item (description wraps, price at end)

**KEY RULE:** Count the PRICES on the right side. Each price = one item!

**How to identify separate items:**
1. Look for prices aligned on RIGHT side of receipt
2. Each distinct price amount = one item
3. Text lines WITHOUT price = part of item above/below
4. Example: If you see TWO prices (12.49 and 1.99), that's TWO items!

**CRITICAL:** 
- Don't combine multiple items into one!
- If receipt shows "6PC ALC" with £12.49 and "MINI FLT" with £1.99 → TWO items
- Each product line with its own price = separate item

- CRITICAL: Check the receipt carefully for ALL items
- Count items by counting DISTINCT PRICES on right side
- Extract EVERY line that has a price on the right side

VAT DETECTION:
- Look for "VAT No" or "VAT#" or "VAT Registration"
- Note if VAT number exists even if no VAT breakdown shown
- Include vat_number field if found

JSON FORMAT (return ONLY this JSON):
{{
  "supplier_name": "full store name exactly as shown",
  "invoice_date": "YYYY-MM-DD (VERIFY THE YEAR CAREFULLY)",
  "invoice_number": "order/receipt number",
  "vat_number": "VAT registration number if shown, else null",
  "gross_total": total amount as number only,
  "vat_amount": VAT amount if separately shown, else null,
  "net_total": net amount if shown, else null,
  "items": [
    {{
      "description": "item name",
      "quantity": quantity as number,
      "unit_price": price per unit if shown,
      "total_price": total for this item
    }}
  ],
  "payment_method": "cash/card/etc if shown",
  "notes": "only useful info, skip 'served by' messages"
}}

Return ONLY the JSON, no additional text or markdown.
"""
        
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        invoice_data = json.loads(content)
        
        # Clean up supplier name
        if invoice_data.get('supplier_name'):
            invoice_data['supplier_name'] = clean_supplier_name(invoice_data['supplier_name'])
        
        # Calculate VAT if missing but VAT number present
        gross = float(invoice_data.get('gross_total', 0) or 0)
        vat = float(invoice_data.get('vat_amount', 0) or 0)
        net = float(invoice_data.get('net_total', 0) or 0)
        has_vat_number = invoice_data.get('vat_number') is not None
        
        # If VAT registered but no VAT shown, calculate 20%
        if has_vat_number and gross > 0 and vat == 0:
            invoice_data['net_total'] = round(gross / 1.2, 2)
            invoice_data['vat_amount'] = round(gross - (gross / 1.2), 2)
        elif gross > 0 and vat > 0 and net == 0:
            invoice_data['net_total'] = round(gross - vat, 2)
        elif gross > 0 and net > 0 and vat == 0:
            invoice_data['vat_amount'] = round(gross - net, 2)
        
        return invoice_data
        
    except Exception as e:
        logger.error(f"Error extracting invoice data: {e}")
        return None

def create_date_confirm_keyboard():
    """Create keyboard for date confirmation."""
    keyboard = [
        [
            InlineKeyboardButton("✅ Date Correct", callback_data="date_correct"),
            InlineKeyboardButton("✏️ Edit Date", callback_data="date_edit")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_branch_keyboard():
    """Create inline keyboard for branch selection."""
    keyboard = []
    for i in range(0, len(BRANCHES), 2):
        row = [InlineKeyboardButton(BRANCHES[i], callback_data=f"branch_{BRANCHES[i]}")]
        if i + 1 < len(BRANCHES):
            row.append(InlineKeyboardButton(BRANCHES[i+1], callback_data=f"branch_{BRANCHES[i+1]}"))
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)

def create_category_keyboard():
    """Create inline keyboard for category selection."""
    keyboard = []
    for i in range(0, len(CATEGORIES), 2):
        row = [InlineKeyboardButton(CATEGORIES[i], callback_data=f"category_{CATEGORIES[i]}")]
        if i + 1 < len(CATEGORIES):
            row.append(InlineKeyboardButton(CATEGORIES[i+1], callback_data=f"category_{CATEGORIES[i+1]}"))
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)

def create_vat_keyboard():
    """Create inline keyboard for VAT confirmation."""
    keyboard = [
        [
            InlineKeyboardButton("✅ VAT Correct", callback_data="vat_correct"),
            InlineKeyboardButton("❌ No VAT (0%)", callback_data="vat_zero")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_edit_keyboard():
    """Create inline keyboard for edit menu."""
    keyboard = [
        [InlineKeyboardButton("✏️ Edit Supplier", callback_data="edit_supplier")],
        [InlineKeyboardButton("📅 Edit Date", callback_data="edit_date")],
        [InlineKeyboardButton("🔢 Edit Invoice Number", callback_data="edit_invoice_num")],
        [InlineKeyboardButton("💰 Edit Amount", callback_data="edit_amount")],
        [InlineKeyboardButton("📦 Edit Items", callback_data="edit_items")],
        [InlineKeyboardButton("🔙 Back to Review", callback_data="edit_back")],
        [InlineKeyboardButton("❌ Cancel", callback_data="edit_cancel")]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_items_keyboard(items):
    """Create keyboard for item management."""
    keyboard = []
    
    # Show each item with edit/delete buttons
    for idx, item in enumerate(items[:10], 1):  # Max 10 items shown
        desc = item.get('description', 'Unknown')[:30]  # Truncate long names
        keyboard.append([InlineKeyboardButton(
            f"{idx}. {desc}",
            callback_data=f"item_view_{idx-1}"
        )])
    
    # Action buttons
    keyboard.append([InlineKeyboardButton("➕ Add Item", callback_data="item_add")])
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="items_back")])
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="items_cancel")])
    
    return InlineKeyboardMarkup(keyboard)

def create_item_action_keyboard(item_idx):
    """Create keyboard for individual item actions."""
    keyboard = [
        [InlineKeyboardButton("✏️ Edit Description", callback_data=f"itemfield_desc_{item_idx}")],
        [InlineKeyboardButton("🔢 Edit Values", callback_data=f"itemfield_values_{item_idx}")],
        [InlineKeyboardButton("🗑️ Delete Item", callback_data=f"item_delete_{item_idx}")],
        [InlineKeyboardButton("🔙 Back to Items", callback_data="item_back_list")],
        [InlineKeyboardButton("❌ Cancel", callback_data="item_cancel")]
    ]
    return InlineKeyboardMarkup(keyboard)

def create_confirm_keyboard():
    """Create inline keyboard for final confirmation."""
    keyboard = [
        [
            InlineKeyboardButton("✅ Confirm & Save", callback_data="confirm_save"),
            InlineKeyboardButton("✏️ Edit", callback_data="confirm_edit"),
            InlineKeyboardButton("❌ Cancel", callback_data="confirm_cancel")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def format_invoice_preview(invoice_data, branch=None, category=None):
    """Format invoice data for preview."""
    message = "📄 **Invoice Details**\n\n"
    
    message += f"🏪 **Supplier:** {invoice_data.get('supplier_name', 'Unknown')}\n"
    
    date_formatted = format_date(invoice_data.get('invoice_date'))
    message += f"📅 **Date:** {date_formatted}\n"
    
    # Date warning
    date_valid, date_msg = validate_date(invoice_data.get('invoice_date', ''))
    if not date_valid:
        message += f"⚠️ *Warning: {date_msg}*\n"
    
    if invoice_data.get('invoice_number'):
        message += f"🔢 **Invoice #:** {invoice_data.get('invoice_number')}\n"
    
    if invoice_data.get('vat_number'):
        message += f"🆔 **VAT #:** {invoice_data.get('vat_number')}\n"
    
    if branch:
        message += f"🏢 **Branch:** {branch}\n"
    
    if category:
        message += f"📦 **Category:** {category}\n"
    
    message += f"\n💰 **Gross Total:** {format_currency(invoice_data.get('gross_total'))}\n"
    message += f"💵 **VAT (20%):** {format_currency(invoice_data.get('vat_amount'))}\n"
    message += f"💸 **Net Total:** {format_currency(invoice_data.get('net_total'))}\n"
    
    if invoice_data.get('payment_method'):
        message += f"💳 **Payment:** {invoice_data.get('payment_method')}\n"
    
    # Show items
    items = invoice_data.get('items', [])
    if items and len(items) > 0:
        message += f"\n📦 **Items ({len(items)}):**\n"
        for idx, item in enumerate(items[:5], 1):
            desc = item.get('description', 'Unknown')
            qty = item.get('quantity', 1)
            price = format_currency(item.get('total_price', item.get('unit_price', 0)))
            if qty and float(qty) != 1.0:
                message += f"{idx}. {desc} (x{qty}) - {price}\n"
            else:
                message += f"{idx}. {desc} - {price}\n"
        
        if len(items) > 5:
            message += f"... and {len(items) - 5} more items\n"
    
    if invoice_data.get('notes') and invoice_data.get('notes').strip():
        notes = invoice_data.get('notes').strip()
        if len(notes) < 100 and 'served by' not in notes.lower() and 'administrator' not in notes.lower():
            message += f"\n📝 **Notes:** {notes}\n"
    
    return message

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message."""
    user = update.effective_user
    welcome_message = f"""
👋 Welcome {user.first_name} to Kochi Bay Invoice Bot!

📸 **How to use:**
1. Send me a photo of your invoice
2. Confirm the date (or edit if wrong)
3. Select branch and category
4. Review and confirm
5. Done! Invoice saved instantly

💡 **Commands:**
/start - Show this message
/help - Get help
/mystats - Your stats
/recent - Recent invoices

Send me an invoice photo to start! 🚀
"""
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message."""
    help_text = """
📚 **Kochi Bay Invoice Bot Help**

**Sending Invoices:**
• Send clear photo of invoice
• Confirm or edit the date
• Select branch and category using buttons
• Review and confirm
• Saved automatically!

**Managing Your Invoices:**
• /myinvoices - View your last 10 invoices
• /edit [number] - Edit invoice (e.g. /edit 123)
• /delete [number] - Delete invoice (e.g. /delete 123)

**Tips:**
✅ Good lighting
✅ Receipt flat
✅ All text visible
✅ Capture full receipt

**Commands:**
/start - Welcome
/help - This help
/mystats - Your statistics
/recent - Last 5 invoices
/myinvoices - Your invoices (with edit/delete)
/cancel - Cancel submission
"""
    await update.message.reply_text(help_text)

async def my_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user statistics."""
    user = update.effective_user
    telegram_username = user.username or user.first_name
    
    try:
        response = supabase.table('invoices').select('*').eq('created_by', telegram_username).execute()
        
        total_invoices = len(response.data)
        
        if total_invoices == 0:
            await update.message.reply_text("📊 You haven't submitted any invoices yet!")
            return
        
        total_amount = sum(float(inv.get('gross_total', 0) or 0) for inv in response.data)
        recent = response.data[0] if response.data else None
        recent_date = recent.get('invoice_date', 'N/A') if recent else 'N/A'
        
        stats_message = f"""
📊 **Your Invoice Statistics**

👤 User: {telegram_username}
📝 Total Invoices: {total_invoices}
💰 Total Amount: {format_currency(total_amount)}
📅 Last Submission: {format_date(recent_date)}

Keep up the great work! 🎉
"""
        await update.message.reply_text(stats_message)
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        await update.message.reply_text("❌ Error fetching statistics.")

async def recent_invoices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent invoices."""
    try:
        response = supabase.table('invoices').select('*').order('created_at', desc=True).limit(5).execute()
        
        if not response.data:
            await update.message.reply_text("📭 No invoices found.")
            return
        
        message = "📋 **Recent Invoices (Last 5):**\n\n"
        
        for idx, inv in enumerate(response.data, 1):
            entry_num = inv.get('entry_number', 'N/A')
            supplier = inv.get('supplier', 'Unknown')
            amount = format_currency(inv.get('gross_total', 0))
            date = format_date(inv.get('invoice_date', 'N/A'))
            created_by = inv.get('created_by', 'Unknown')
            branch = inv.get('branch', 'N/A')
            
            message += f"{idx}. **Entry #{entry_num}** - {supplier}\n"
            message += f"   💰 {amount} | 📅 {date}\n"
            message += f"   🏢 {branch} | 👤 {created_by}\n\n"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Error fetching recent invoices: {e}")
        await update.message.reply_text("❌ Error fetching invoices.")

async def my_invoices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user's invoices with edit/delete options."""
    user = update.effective_user
    telegram_username = staff_names.get(user.id, user.username or user.first_name)
    
    try:
        response = supabase.table('invoices').select('*').eq('created_by', telegram_username).order('created_at', desc=True).limit(10).execute()
        
        if not response.data:
            await update.message.reply_text("📭 You haven't submitted any invoices yet!")
            return
        
        message = f"📋 **Your Invoices (Last 10):**\n\n"
        
        for idx, inv in enumerate(response.data, 1):
            entry_num = inv.get('entry_number', 'N/A')
            supplier = inv.get('supplier', 'Unknown')
            amount = format_currency(inv.get('gross_total', 0))
            date = format_date(inv.get('invoice_date', 'N/A'))
            branch = inv.get('branch', 'N/A')
            
            message += f"{idx}. **Entry #{entry_num}** - {supplier}\n"
            message += f"   💰 {amount} | 📅 {date} | 🏢 {branch}\n"
        
        message += f"\n**To manage:**\n"
        message += f"• Edit: `/edit {response.data[0].get('entry_number')}`\n"
        message += f"• Delete: `/delete {response.data[0].get('entry_number')}`"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Error fetching user invoices: {e}")
        await update.message.reply_text("❌ Error fetching your invoices.")

async def edit_invoice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Edit an invoice by entry number."""
    user = update.effective_user
    telegram_username = staff_names.get(user.id, user.username or user.first_name)
    
    # Get entry number from command
    if not context.args or len(context.args) == 0:
        await update.message.reply_text(
            "❌ Please provide entry number\n\n"
            "Usage: `/edit 123`\n\n"
            "Use /myinvoices to see your entries"
        )
        return
    
    try:
        entry_number = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid entry number. Must be a number.")
        return
    
    try:
        # Fetch invoice
        response = supabase.table('invoices').select('*').eq('entry_number', entry_number).eq('created_by', telegram_username).execute()
        
        if not response.data or len(response.data) == 0:
            await update.message.reply_text(
                f"❌ Entry #{entry_number} not found or not created by you.\n\n"
                "Use /myinvoices to see your entries"
            )
            return
        
        invoice = response.data[0]
        
        # Show current details
        date_formatted = format_date(invoice.get('invoice_date'))
        items = invoice.get('line_items', [])
        
        message = f"📝 **Edit Entry #{entry_number}**\n\n"
        message += f"**Current Details:**\n"
        message += f"🏪 Supplier: {invoice.get('supplier')}\n"
        message += f"📅 Date: {date_formatted}\n"
        message += f"🏢 Branch: {invoice.get('branch')}\n"
        message += f"📦 Category: {invoice.get('category')}\n"
        message += f"💰 Gross: {format_currency(invoice.get('gross_total'))}\n"
        message += f"\n**Items ({len(items)}):**\n"
        for idx, item in enumerate(items[:5], 1):
            message += f"{idx}. {item.get('itemName')} - {format_currency(item.get('rate'))}\n"
        
        message += f"\n⚠️ **Edit via Web Interface**\n"
        message += f"For now, please edit via: https://kbetracker.netlify.app\n"
        message += f"Full bot editing coming soon!"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Error editing invoice: {e}")
        await update.message.reply_text("❌ Error loading invoice.")

async def delete_invoice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete an invoice by entry number."""
    user = update.effective_user
    telegram_username = staff_names.get(user.id, user.username or user.first_name)
    
    # Get entry number from command
    if not context.args or len(context.args) == 0:
        await update.message.reply_text(
            "❌ Please provide entry number\n\n"
            "Usage: `/delete 123`\n\n"
            "Use /myinvoices to see your entries"
        )
        return
    
    try:
        entry_number = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid entry number. Must be a number.")
        return
    
    try:
        # Fetch invoice
        response = supabase.table('invoices').select('*').eq('entry_number', entry_number).eq('created_by', telegram_username).execute()
        
        if not response.data or len(response.data) == 0:
            await update.message.reply_text(
                f"❌ Entry #{entry_number} not found or not created by you.\n\n"
                "Use /myinvoices to see your entries"
            )
            return
        
        invoice = response.data[0]
        invoice_id = invoice.get('id')
        
        # Show details and ask for confirmation
        date_formatted = format_date(invoice.get('invoice_date'))
        
        message = f"⚠️ **Confirm Deletion**\n\n"
        message += f"Entry #{entry_number}\n"
        message += f"🏪 {invoice.get('supplier')}\n"
        message += f"📅 {date_formatted}\n"
        message += f"💰 {format_currency(invoice.get('gross_total'))}\n\n"
        message += f"**To confirm, type:** `/confirmdelete {entry_number}`\n"
        message += f"**To cancel:** Just ignore this message"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Error deleting invoice: {e}")
        await update.message.reply_text("❌ Error loading invoice.")

async def confirm_delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm and execute invoice deletion."""
    user = update.effective_user
    telegram_username = staff_names.get(user.id, user.username or user.first_name)
    
    if not context.args or len(context.args) == 0:
        await update.message.reply_text("❌ Please provide entry number to confirm deletion")
        return
    
    try:
        entry_number = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid entry number.")
        return
    
    try:
        # Fetch and delete invoice
        response = supabase.table('invoices').select('*').eq('entry_number', entry_number).eq('created_by', telegram_username).execute()
        
        if not response.data or len(response.data) == 0:
            await update.message.reply_text(f"❌ Entry #{entry_number} not found.")
            return
        
        invoice_id = response.data[0].get('id')
        
        # Delete
        delete_response = supabase.table('invoices').delete().eq('id', invoice_id).execute()
        
        await update.message.reply_text(
            f"✅ **Entry #{entry_number} deleted successfully!**\n\n"
            f"Invoice removed from system."
        )
        
    except Exception as e:
        logger.error(f"Error confirming delete: {e}")
        await update.message.reply_text("❌ Error deleting invoice.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle photo messages - start of flow."""
    user = update.effective_user
    
    # Check if staff name is stored
    if user.id not in staff_names:
        await update.message.reply_text(
            "👤 **Welcome!**\n\n"
            "Before we start, please enter your full name for accountability:\n\n"
            "Example: John Smith"
        )
        # Store the photo for later
        photo = update.message.photo[-1]
        pending_invoices[user.id] = {'photo_file_id': photo.file_id}
        return ASK_STAFF_NAME
    
    telegram_username = staff_names.get(user.id, user.username or user.first_name)
    
    await update.message.reply_text("📸 Processing your invoice... Please wait.")
    
    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()
        
        invoice_data = await extract_invoice_data(bytes(image_bytes))
        
        if not invoice_data:
            await update.message.reply_text("❌ Failed to extract invoice data. Please try again with a clearer photo.")
            return ConversationHandler.END
        
        # Store data
        pending_invoices[user.id] = {
            'data': invoice_data,
            'telegram_user': telegram_username,
            'photo_file_id': photo.file_id
        }
        
        # Show date confirmation
        date_formatted = format_date(invoice_data.get('invoice_date'))
        date_valid, date_msg = validate_date(invoice_data.get('invoice_date', ''))
        
        message = f"📅 **Extracted Date:** {date_formatted}\n\n"
        if not date_valid:
            message += f"⚠️ Warning: {date_msg}\n\n"
        message += "Is this date correct?"
        
        await update.message.reply_text(
            message,
            reply_markup=create_date_confirm_keyboard()
        )
        
        return CONFIRM_DATE
        
    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await update.message.reply_text("❌ Error processing invoice. Please try again.")
        return ConversationHandler.END

async def handle_staff_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle staff name input."""
    user = update.effective_user
    staff_name = update.message.text.strip()
    
    if not staff_name or len(staff_name) < 2:
        await update.message.reply_text("❌ Please enter a valid name (at least 2 characters).")
        return ASK_STAFF_NAME
    
    # Store staff name
    staff_names[user.id] = staff_name
    
    await update.message.reply_text(f"✅ Thank you, {staff_name}!\n\n📸 Processing your invoice...")
    
    # Now process the photo that was sent earlier
    if user.id in pending_invoices and 'photo_file_id' in pending_invoices[user.id]:
        photo_file_id = pending_invoices[user.id]['photo_file_id']
        file = await context.bot.get_file(photo_file_id)
        image_bytes = await file.download_as_bytearray()
        
        invoice_data = await extract_invoice_data(bytes(image_bytes))
        
        if not invoice_data:
            await update.message.reply_text("❌ Failed to extract invoice data. Please send the photo again.")
            return ConversationHandler.END
        
        pending_invoices[user.id] = {
            'data': invoice_data,
            'telegram_user': staff_name,
            'photo_file_id': photo_file_id
        }
        
        date_formatted = format_date(invoice_data.get('invoice_date'))
        date_valid, date_msg = validate_date(invoice_data.get('invoice_date', ''))
        
        message = f"📅 **Extracted Date:** {date_formatted}\n\n"
        if not date_valid:
            message += f"⚠️ Warning: {date_msg}\n\n"
        message += "Is this date correct?"
        
        await update.message.reply_text(
            message,
            reply_markup=create_date_confirm_keyboard()
        )
        
        return CONFIRM_DATE
    
    await update.message.reply_text("❌ Error: Photo not found. Please send the invoice photo again.")
    return ConversationHandler.END

async def handle_date_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle date confirmation button."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    if query.data == "date_edit":
        await query.edit_message_text(
            "📅 Please enter the correct date.\n\n"
            "Format: DD/MM/YYYY\n"
            "Example: 24/12/2025"
        )
        return EDIT_DATE
    
    # Date confirmed, move to branch selection
    await query.edit_message_text(
        "✅ Date confirmed\n\n🏢 **Select Branch:**",
        reply_markup=create_branch_keyboard()
    )
    
    return SELECT_BRANCH

async def handle_date_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle manual date input."""
    user = update.effective_user
    date_input = update.message.text.strip()
    
    # Allow cancel
    if date_input.lower() in ['cancel', 'back']:
        # Check if we're in edit menu or initial flow
        if user.id in pending_invoices and 'branch' in pending_invoices[user.id]:
            # In edit menu
            await update.message.reply_text(
                "❌ Cancelled\n\n✏️ **Edit Invoice Details**\n\nWhat else?",
                reply_markup=create_edit_keyboard()
            )
            return EDIT_MENU
        else:
            # In initial flow, go to branch selection
            await update.message.reply_text(
                "❌ Cancelled\n\n🏢 **Select Branch:**",
                reply_markup=create_branch_keyboard()
            )
            return SELECT_BRANCH
    
    if user.id not in pending_invoices:
        await update.message.reply_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    # Parse date
    parsed_date = parse_date_input(date_input)
    
    if not parsed_date:
        await update.message.reply_text(
            "❌ Invalid date format. Please use DD/MM/YYYY\n"
            "Example: 24/12/2025\n"
            "Or type 'cancel' to go back"
        )
        return EDIT_DATE
    
    # Update date
    pending_invoices[user.id]['data']['invoice_date'] = parsed_date
    
    # Check if we're in edit menu or initial flow
    if 'branch' in pending_invoices[user.id]:
        # In edit menu
        date_formatted = format_date(parsed_date)
        await update.message.reply_text(
            f"✅ Date updated to: {date_formatted}\n\nWhat else?",
            reply_markup=create_edit_keyboard()
        )
        return EDIT_MENU
    else:
        # In initial flow
        date_formatted = format_date(parsed_date)
        await update.message.reply_text(
            f"✅ Date updated to: {date_formatted}\n\n🏢 **Select Branch:**",
            reply_markup=create_branch_keyboard()
        )
        return SELECT_BRANCH

async def handle_branch_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle branch selection button."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    branch = query.data.replace("branch_", "")
    
    if branch == "Other":
        await query.edit_message_text("🏢 Please type the branch name:")
        return ENTER_OTHER_BRANCH
    
    pending_invoices[user.id]['branch'] = branch
    
    await query.edit_message_text(
        f"✅ Branch: {branch}\n\n📦 **Select Category:**",
        reply_markup=create_category_keyboard()
    )
    
    return SELECT_CATEGORY

async def handle_other_branch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle custom branch input."""
    user = update.effective_user
    branch = update.message.text.strip()
    
    if user.id not in pending_invoices:
        await update.message.reply_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    pending_invoices[user.id]['branch'] = branch
    
    await update.message.reply_text(
        f"✅ Branch: {branch}\n\n📦 **Select Category:**",
        reply_markup=create_category_keyboard()
    )
    
    return SELECT_CATEGORY

async def handle_category_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle category selection button."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    category = query.data.replace("category_", "")
    
    if category == "Other":
        await query.edit_message_text("📦 Please type the category name:")
        return ENTER_OTHER_CATEGORY
    
    pending_invoices[user.id]['category'] = category
    
    # Check VAT
    invoice_data = pending_invoices[user.id]['data']
    vat = float(invoice_data.get('vat_amount', 0) or 0)
    
    if vat > 0:
        await query.edit_message_text(
            f"✅ Category: {category}\n\n💵 **VAT detected: {format_currency(vat)}**\n\nIs this correct?",
            reply_markup=create_vat_keyboard()
        )
        return CONFIRM_VAT
    else:
        return await show_final_confirmation(query, user)

async def handle_other_category(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle custom category input."""
    user = update.effective_user
    category = update.message.text.strip()
    
    if user.id not in pending_invoices:
        await update.message.reply_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    pending_invoices[user.id]['category'] = category
    
    invoice_data = pending_invoices[user.id]['data']
    vat = float(invoice_data.get('vat_amount', 0) or 0)
    
    if vat > 0:
        await update.message.reply_text(
            f"✅ Category: {category}\n\n💵 **VAT detected: {format_currency(vat)}**\n\nIs this correct?",
            reply_markup=create_vat_keyboard()
        )
        return CONFIRM_VAT
    else:
        return await show_final_confirmation_message(update.message, user)

async def handle_vat_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle VAT confirmation button."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    if query.data == "vat_zero":
        pending_invoices[user.id]['data']['vat_amount'] = 0
        pending_invoices[user.id]['data']['net_total'] = pending_invoices[user.id]['data']['gross_total']
    
    return await show_final_confirmation(query, user)

async def show_final_confirmation(query, user) -> int:
    """Show final confirmation with all details."""
    pending = pending_invoices[user.id]
    invoice_data = pending['data']
    branch = pending.get('branch')
    category = pending.get('category')
    
    preview = format_invoice_preview(invoice_data, branch, category)
    
    await query.edit_message_text(
        preview + "\n\n**Please review and confirm:**",
        reply_markup=create_confirm_keyboard(),
        parse_mode='Markdown'
    )
    
    return CONFIRM_DATA

async def show_final_confirmation_message(message, user) -> int:
    """Show final confirmation via message."""
    pending = pending_invoices[user.id]
    invoice_data = pending['data']
    branch = pending.get('branch')
    category = pending.get('category')
    
    preview = format_invoice_preview(invoice_data, branch, category)
    
    await message.reply_text(
        preview + "\n\n**Please review and confirm:**",
        reply_markup=create_confirm_keyboard(),
        parse_mode='Markdown'
    )
    
    return CONFIRM_DATA

async def handle_edit_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show edit menu."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    await query.edit_message_text(
        "✏️ **Edit Invoice Details**\n\nWhat would you like to edit?",
        reply_markup=create_edit_keyboard()
    )
    
    return EDIT_MENU

async def handle_edit_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle edit menu selection."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired. Please send invoice photo again.")
        return ConversationHandler.END
    
    if query.data == "edit_cancel":
        del pending_invoices[user.id]
        await query.edit_message_text("❌ Invoice cancelled.")
        return ConversationHandler.END
    
    if query.data == "edit_back":
        return await show_final_confirmation(query, user)
    
    if query.data == "edit_supplier":
        await query.edit_message_text("🏪 Please enter the supplier name:")
        return EDIT_SUPPLIER
    
    if query.data == "edit_date":
        await query.edit_message_text(
            "📅 Please enter the date.\n\nFormat: DD/MM/YYYY\nExample: 24/12/2025"
        )
        return EDIT_DATE
    
    if query.data == "edit_invoice_num":
        await query.edit_message_text("🔢 Please enter the invoice number:")
        return EDIT_INVOICE_NUM
    
    if query.data == "edit_amount":
        await query.edit_message_text("💰 Please enter the total amount (numbers only):")
        return EDIT_AMOUNT
    
    if query.data == "edit_items":
        return await handle_items_edit(update, context)
    
    return EDIT_MENU

async def handle_supplier_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle supplier edit."""
    user = update.effective_user
    supplier = update.message.text.strip()
    
    # Allow cancel
    if supplier.lower() in ['cancel', 'back']:
        await update.message.reply_text(
            "❌ Cancelled\n\n✏️ **Edit Invoice Details**\n\nWhat else?",
            reply_markup=create_edit_keyboard()
        )
        return EDIT_MENU
    
    if user.id not in pending_invoices:
        await update.message.reply_text("❌ Session expired.")
        return ConversationHandler.END
    
    pending_invoices[user.id]['data']['supplier_name'] = supplier
    
    await update.message.reply_text(
        f"✅ Supplier updated to: {supplier}\n\nWhat else?",
        reply_markup=create_edit_keyboard()
    )
    
    return EDIT_MENU

async def handle_invoice_num_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle invoice number edit."""
    user = update.effective_user
    invoice_num = update.message.text.strip()
    
    # Allow cancel
    if invoice_num.lower() in ['cancel', 'back']:
        await update.message.reply_text(
            "❌ Cancelled\n\n✏️ **Edit Invoice Details**\n\nWhat else?",
            reply_markup=create_edit_keyboard()
        )
        return EDIT_MENU
    
    if user.id not in pending_invoices:
        await update.message.reply_text("❌ Session expired.")
        return ConversationHandler.END
    
    pending_invoices[user.id]['data']['invoice_number'] = invoice_num
    
    await update.message.reply_text(
        f"✅ Invoice number updated to: {invoice_num}\n\nWhat else?",
        reply_markup=create_edit_keyboard()
    )
    
    return EDIT_MENU

async def handle_amount_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle amount edit."""
    user = update.effective_user
    amount_str = update.message.text.strip()
    
    # Allow cancel
    if amount_str.lower() in ['cancel', 'back']:
        await update.message.reply_text(
            "❌ Cancelled\n\n✏️ **Edit Invoice Details**\n\nWhat else?",
            reply_markup=create_edit_keyboard()
        )
        return EDIT_MENU
    
    if user.id not in pending_invoices:
        await update.message.reply_text("❌ Session expired.")
        return ConversationHandler.END
    
    amount_str = amount_str.replace('£', '').replace(',', '')
    
    try:
        amount = float(amount_str)
        pending_invoices[user.id]['data']['gross_total'] = amount
        
        # Recalculate VAT
        vat = round(amount - (amount / 1.2), 2)
        net = round(amount / 1.2, 2)
        pending_invoices[user.id]['data']['vat_amount'] = vat
        pending_invoices[user.id]['data']['net_total'] = net
        
        await update.message.reply_text(
            f"✅ Amount updated to: {format_currency(amount)}\n"
            f"💵 VAT: {format_currency(vat)}\n"
            f"💸 Net: {format_currency(net)}\n\nWhat else?",
            reply_markup=create_edit_keyboard()
        )
        
        return EDIT_MENU
    except:
        await update.message.reply_text(
            "❌ Invalid amount. Please enter numbers only.\n"
            "Or type 'cancel' to go back"
        )
        return EDIT_AMOUNT

async def handle_items_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show items list for editing."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired.")
        return ConversationHandler.END
    
    items = pending_invoices[user.id]['data'].get('items', [])
    
    if not items or len(items) == 0:
        message = "📦 **No items found!**\n\nWould you like to add items manually?"
        keyboard = [
            [InlineKeyboardButton("➕ Add Item", callback_data="item_add")],
            [InlineKeyboardButton("🔙 Back", callback_data="items_back")]
        ]
        await query.edit_message_text(message, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        message = f"📦 **Edit Items** ({len(items)} items)\n\nSelect an item to edit or delete:"
        await query.edit_message_text(
            message,
            reply_markup=create_items_keyboard(items)
        )
    
    return EDIT_ITEMS

async def handle_item_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle item selection from list."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired.")
        return ConversationHandler.END
    
    if query.data == "items_cancel":
        await query.edit_message_text(
            "✏️ **Edit Invoice Details**\n\nWhat would you like to edit?",
            reply_markup=create_edit_keyboard()
        )
        return EDIT_MENU
    
    if query.data == "items_back":
        await query.edit_message_text(
            "✏️ **Edit Invoice Details**\n\nWhat would you like to edit?",
            reply_markup=create_edit_keyboard()
        )
        return EDIT_MENU
    
    if query.data == "item_add":
        pending_invoices[user.id]['editing_item'] = 'new'
        await query.edit_message_text(
            "➕ **Add New Item**\n\n"
            "**Format:** `Name, Rate, Qty, UOM, VAT%`\n"
            "(Bot auto-calculates Net, VAT Amt, Gross)\n\n"
            "**Example 1** (VAT included in rate):\n"
            "`6PC ALC, 12.49, 1, ea, 20`\n"
            "→ Net: £10.41, VAT: £2.08, Gross: £12.49\n\n"
            "**Example 2** (No VAT):\n"
            "`Milk, 1.50, 2, ltr, 0`\n"
            "→ Net: £3.00, VAT: £0.00, Gross: £3.00\n\n"
            "**Example 3** (Fresh produce with VAT):\n"
            "`Raw Mango, 7.99, 1.935, kg, 20`\n\n"
            "Type 'cancel' or 'back' to return"
        )
        return ADD_ITEM
    
    if query.data.startswith("item_view_"):
        item_idx = int(query.data.replace("item_view_", ""))
        items = pending_invoices[user.id]['data'].get('items', [])
        
        if item_idx < len(items):
            item = items[item_idx]
            message = (
                f"📦 **Item {item_idx + 1}**\n\n"
                f"**Description:** {item.get('description', 'N/A')}\n"
                f"**Quantity:** {item.get('quantity', 1)} {item.get('unit', '')}\n"
                f"**Rate:** {format_currency(item.get('unit_price', 0))}\n"
                f"**VAT%:** {item.get('vat_percentage', 0)}%\n"
                f"**Net:** {format_currency(item.get('net_amount', 0))}\n"
                f"**VAT Amt:** {format_currency(item.get('vat_amount', 0))}\n"
                f"**Gross:** {format_currency(item.get('total_price', 0))}\n\n"
                f"What would you like to do?"
            )
            await query.edit_message_text(
                message,
                reply_markup=create_item_action_keyboard(item_idx)
            )
            return EDIT_ITEMS_ACTION
    
    if query.data == "item_back_list":
        items = pending_invoices[user.id]['data'].get('items', [])
        message = f"📦 **Edit Items** ({len(items)} items)\n\nSelect an item:"
        await query.edit_message_text(
            message,
            reply_markup=create_items_keyboard(items)
        )
        return EDIT_ITEMS
    
    return EDIT_ITEMS

async def handle_item_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle item edit/delete actions."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired.")
        return ConversationHandler.END
    
    if query.data == "item_back_list":
        items = pending_invoices[user.id]['data'].get('items', [])
        message = f"📦 **Edit Items** ({len(items)} items)\n\nSelect an item:"
        await query.edit_message_text(
            message,
            reply_markup=create_items_keyboard(items)
        )
        return EDIT_ITEMS
    
    if query.data == "item_cancel":
        items = pending_invoices[user.id]['data'].get('items', [])
        await query.edit_message_text(
            f"📦 **Edit Items** ({len(items)} items)\n\nSelect an item:",
            reply_markup=create_items_keyboard(items)
        )
        return EDIT_ITEMS
    
    if query.data.startswith("item_delete_"):
        item_idx = int(query.data.replace("item_delete_", ""))
        items = pending_invoices[user.id]['data'].get('items', [])
        
        if item_idx < len(items):
            deleted_item = items.pop(item_idx)
            pending_invoices[user.id]['data']['items'] = items
            
            if len(items) > 0:
                await query.edit_message_text(
                    f"✅ Item deleted: {deleted_item.get('description')}\n\n"
                    f"Remaining items: {len(items)}",
                    reply_markup=create_items_keyboard(items)
                )
            else:
                keyboard = [
                    [InlineKeyboardButton("➕ Add Item", callback_data="item_add")],
                    [InlineKeyboardButton("🔙 Back", callback_data="items_back")],
                    [InlineKeyboardButton("❌ Cancel", callback_data="items_cancel")]
                ]
                await query.edit_message_text(
                    f"✅ Item deleted: {deleted_item.get('description')}\n\n"
                    f"No items remaining.",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
        return EDIT_ITEMS
    
    if query.data.startswith("itemfield_"):
        parts = query.data.replace("itemfield_", "").split("_")
        field_type = parts[0]
        item_idx = int(parts[1])
        
        pending_invoices[user.id]['editing_item'] = item_idx
        pending_invoices[user.id]['editing_field'] = field_type
        
        if field_type == "desc":
            await query.edit_message_text(
                "✏️ Enter new description:\n\n"
                "Type 'cancel' or 'back' to return"
            )
        elif field_type == "values":
            await query.edit_message_text(
                "🔢 Enter new values:\n\n"
                "**Format:** `Rate, Qty, UOM, VAT%`\n"
                "(Bot auto-calculates Net, VAT, Gross)\n\n"
                "**Example:**\n"
                "`12.49, 1, ea, 20`\n\n"
                "Type 'cancel' or 'back' to return"
            )
        
        return EDIT_ITEM_FIELD
    
    return EDIT_ITEMS_ACTION

async def handle_add_item(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle adding new item with format: Name, Rate, Qty, UOM, VAT% (auto-calculates amounts)."""
    user = update.effective_user
    text = update.message.text.strip()
    
    # Allow cancel
    if text.lower() in ['cancel', 'back']:
        items = pending_invoices[user.id]['data'].get('items', [])
        await update.message.reply_text(
            f"❌ Cancelled\n\n📦 **Edit Items** ({len(items)} items)",
            reply_markup=create_items_keyboard(items)
        )
        return EDIT_ITEMS
    
    if user.id not in pending_invoices:
        await update.message.reply_text("❌ Session expired.")
        return ConversationHandler.END
    
    # Parse: Name, Rate, Qty, UOM, VAT% (simplified!)
    parts = [p.strip() for p in text.split(',')]
    
    if len(parts) < 4:
        await update.message.reply_text(
            "❌ Invalid format. Please use:\n"
            "`Name, Rate, Qty, UOM, VAT%`\n\n"
            "**Example 1** (VAT included in rate):\n"
            "`6PC ALC, 12.49, 1, ea, 20`\n"
            "Bot calculates: Net=10.41, VAT=2.08, Gross=12.49\n\n"
            "**Example 2** (Rate is net):\n"
            "`Chicken, 7.49, 0.5, kg, 20`\n"
            "Bot calculates: Net=3.75, VAT=0.75, Gross=4.50\n\n"
            "Or type 'cancel' to go back"
        )
        return ADD_ITEM
    
    try:
        name = parts[0]
        rate = float(parts[1])
        qty = float(parts[2])
        uom = parts[3] if len(parts) > 3 else 'ea'
        vat_pct = float(parts[4]) if len(parts) > 4 else 0
        
        # Smart calculation: Assume rate INCLUDES VAT if VAT% > 0
        if vat_pct > 0:
            # Rate includes VAT, calculate backwards
            gross_per_unit = rate
            net_per_unit = rate / (1 + vat_pct / 100)
            vat_per_unit = gross_per_unit - net_per_unit
            
            net_amt = net_per_unit * qty
            vat_amt = vat_per_unit * qty
            gross_amt = gross_per_unit * qty
        else:
            # No VAT
            net_amt = rate * qty
            vat_amt = 0
            gross_amt = net_amt
        
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid numbers. Please check your values.\n\n"
            "Format: `Name, Rate, Qty, UOM, VAT%`\n"
            "Example: `6PC ALC, 12.49, 1, ea, 20`\n\n"
            "Or type 'cancel' to go back"
        )
        return ADD_ITEM
    
    # Create item with all calculated fields
    new_item = {
        'description': name,
        'quantity': qty,
        'unit': uom,
        'unit_price': rate,
        'vat_percentage': vat_pct,
        'net_amount': net_amt,
        'vat_amount': vat_amt,
        'total_price': gross_amt
    }
    
    items = pending_invoices[user.id]['data'].get('items', [])
    items.append(new_item)
    pending_invoices[user.id]['data']['items'] = items
    
    await update.message.reply_text(
        f"✅ **Item added:** {name}\n"
        f"Rate: £{rate:.2f} | Qty: {qty} {uom} | VAT: {vat_pct}%\n\n"
        f"**Calculated:**\n"
        f"Net: £{net_amt:.2f}\n"
        f"VAT: £{vat_amt:.2f}\n"
        f"Gross: £{gross_amt:.2f}\n\n"
        f"**Total items:** {len(items)}",
        reply_markup=create_items_keyboard(items)
    )
    
    return EDIT_ITEMS

async def handle_item_field_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle editing specific item field with auto-calculation."""
    user = update.effective_user
    text = update.message.text.strip()
    
    # Allow cancel
    if text.lower() in ['cancel', 'back']:
        item_idx = pending_invoices[user.id].get('editing_item', 0)
        items = pending_invoices[user.id]['data'].get('items', [])
        if item_idx < len(items):
            item = items[item_idx]
            await update.message.reply_text(
                f"❌ Cancelled\n\n📦 **Item {item_idx + 1}**\n{item.get('description')}",
                reply_markup=create_item_action_keyboard(item_idx)
            )
            return EDIT_ITEMS_ACTION
        else:
            await update.message.reply_text(
                "❌ Cancelled",
                reply_markup=create_items_keyboard(items)
            )
            return EDIT_ITEMS
    
    if user.id not in pending_invoices:
        await update.message.reply_text("❌ Session expired.")
        return ConversationHandler.END
    
    item_idx = pending_invoices[user.id].get('editing_item')
    field_type = pending_invoices[user.id].get('editing_field')
    items = pending_invoices[user.id]['data'].get('items', [])
    
    if item_idx >= len(items):
        await update.message.reply_text("❌ Item not found.")
        return EDIT_ITEMS
    
    item = items[item_idx]
    
    try:
        if field_type == "desc":
            item['description'] = text
            await update.message.reply_text(
                f"✅ Description updated to: {text}",
                reply_markup=create_item_action_keyboard(item_idx)
            )
        elif field_type == "values":
            # Parse: Rate, Qty, UOM, VAT%
            parts = [p.strip() for p in text.split(',')]
            if len(parts) < 3:
                await update.message.reply_text(
                    "❌ Need at least: Rate, Qty, UOM\n"
                    "Format: `Rate, Qty, UOM, VAT%`\n"
                    "Example: `12.49, 1, ea, 20`\n\n"
                    "Type 'cancel' to go back"
                )
                return EDIT_ITEM_FIELD
            
            rate = float(parts[0])
            qty = float(parts[1])
            uom = parts[2]
            vat_pct = float(parts[3]) if len(parts) > 3 else 0
            
            # Smart calculation: Assume rate INCLUDES VAT if VAT% > 0
            if vat_pct > 0:
                # Rate includes VAT, calculate backwards
                gross_per_unit = rate
                net_per_unit = rate / (1 + vat_pct / 100)
                vat_per_unit = gross_per_unit - net_per_unit
                
                net_amt = net_per_unit * qty
                vat_amt = vat_per_unit * qty
                gross_amt = gross_per_unit * qty
            else:
                # No VAT
                net_amt = rate * qty
                vat_amt = 0
                gross_amt = net_amt
            
            item['unit_price'] = rate
            item['quantity'] = qty
            item['unit'] = uom
            item['vat_percentage'] = vat_pct
            item['net_amount'] = net_amt
            item['vat_amount'] = vat_amt
            item['total_price'] = gross_amt
            
            await update.message.reply_text(
                f"✅ Values updated!\n"
                f"Rate: £{rate:.2f} | Qty: {qty} {uom} | VAT: {vat_pct}%\n\n"
                f"**Calculated:**\n"
                f"Net: £{net_amt:.2f}\n"
                f"VAT: £{vat_amt:.2f}\n"
                f"Gross: £{gross_amt:.2f}",
                reply_markup=create_item_action_keyboard(item_idx)
            )
        
        items[item_idx] = item
        pending_invoices[user.id]['data']['items'] = items
        
    except ValueError:
        await update.message.reply_text(
            "❌ Invalid input. Please enter valid numbers.\n"
            "Format: `Rate, Qty, UOM, VAT%`\n"
            "Example: `12.49, 1, ea, 20`\n\n"
            "Type 'cancel' to go back"
        )
        return EDIT_ITEM_FIELD
    
    return EDIT_ITEMS_ACTION

async def get_next_entry_number():
    """Get and increment entry number."""
    try:
        response = supabase.table('settings').select('value').eq('key', 'next_entry_number').execute()
        
        if not response.data:
            supabase.table('settings').insert({'key': 'next_entry_number', 'value': '1'}).execute()
            current_number = 1
        else:
            current_number = int(response.data[0]['value'])
        
        next_number = current_number + 1
        supabase.table('settings').update({'value': str(next_number)}).eq('key', 'next_entry_number').execute()
        
        return current_number
    except Exception as e:
        logger.error(f"Error getting entry number: {e}")
        return int(datetime.now().timestamp())

async def save_to_supabase(invoice_data, telegram_user, branch, category):
    """Save invoice to Supabase matching HTML field names."""
    try:
        entry_number = await get_next_entry_number()
        
        items = invoice_data.get('items', [])
        line_items = []
        
        # Calculate totals from items
        total_net = 0
        total_vat = 0
        total_gross = 0
        
        for item in items:
            # Map bot fields to HTML field names
            rate = float(item.get('unit_price', 0) or 0)
            qty = float(item.get('quantity', 1) or 1)
            vat_pct = float(item.get('vat_percentage', 0) or 0)
            
            # Calculate amounts (matching HTML calculation logic)
            net = rate * qty
            vat = net * (vat_pct / 100)
            gross = net + vat
            
            # Use HTML field names
            item_data = {
                'itemName': item.get('description', ''),  # description → itemName
                'rate': rate,                             # unit_price → rate
                'quantity': qty,
                'uom': item.get('unit', 'kg'),           # unit → uom
                'vatRate': str(int(vat_pct)),            # vat_percentage → vatRate (as string)
                'netAmount': str(round(net, 2))          # Calculated, stored as string
            }
            line_items.append(item_data)
            
            # Add to totals
            total_net += net
            total_vat += vat
            total_gross += gross
        
        # Use calculated totals if items exist
        if len(items) > 0:
            gross_total = total_gross
            vat_total = total_vat
            net_total = total_net
        else:
            gross_total = float(invoice_data.get('gross_total', 0) or 0)
            vat_total = float(invoice_data.get('vat_amount', 0) or 0)
            net_total = float(invoice_data.get('net_total', 0) or 0)
        
        db_data = {
            'entry_number': entry_number,
            'invoice_date': invoice_data.get('invoice_date'),
            'supplier': invoice_data.get('supplier_name', 'Unknown'),
            'branch': get_branch_code(branch),  # Convert to code: Wirral → WR
            'category': category,
            'payment_method': invoice_data.get('payment_method'),
            'payment_reference': None,
            'invoice_number': invoice_data.get('invoice_number'),
            'invoice_type': 'purchase',
            'line_items': line_items if line_items else None,
            'net_total': round(net_total, 2) if net_total else 0,
            'vat_total': round(vat_total, 2) if vat_total else 0,
            'gross_total': round(gross_total, 2) if gross_total else 0,
            'remarks': invoice_data.get('notes'),
            'created_by': telegram_user,
            'created_at': datetime.now().isoformat()
        }
        
        response = supabase.table('invoices').insert(db_data).execute()
        return response.data[0] if response.data else None
        
    except Exception as e:
        logger.error(f"Error saving to Supabase: {e}")
        return None

async def handle_final_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle final confirmation buttons."""
    query = update.callback_query
    await query.answer()
    
    user = query.from_user
    
    if user.id not in pending_invoices:
        await query.edit_message_text("❌ Session expired.")
        return ConversationHandler.END
    
    if query.data == "confirm_cancel":
        del pending_invoices[user.id]
        await query.edit_message_text("❌ Invoice cancelled.")
        return ConversationHandler.END
    
    if query.data == "confirm_edit":
        return await handle_edit_menu(update, context)
    
    if query.data == "confirm_save":
        pending = pending_invoices[user.id]
        invoice_data = pending['data']
        branch = pending.get('branch')
        category = pending.get('category')
        telegram_user = pending['telegram_user']
        
        await query.edit_message_text("💾 Saving invoice...")
        
        saved = await save_to_supabase(invoice_data, telegram_user, branch, category)
        
        if saved:
            date_formatted = format_date(saved.get('invoice_date'))
            await query.message.reply_text(
                f"✅ **Invoice saved successfully!**\n\n"
                f"📊 Entry #: {saved.get('entry_number')}\n"
                f"🏪 Supplier: {saved.get('supplier')}\n"
                f"📅 Date: {date_formatted}\n"
                f"💰 Gross Total: {format_currency(saved.get('gross_total'))}\n"
                f"💵 VAT: {format_currency(saved.get('vat_total'))}\n"
                f"💸 Net Total: {format_currency(saved.get('net_total'))}\n"
                f"🏢 Branch: {saved.get('branch')}\n"
                f"📦 Category: {saved.get('category')}\n\n"
                f"**Thank you! Please submit the physical invoice ASAP** 📄✅"
            )
            del pending_invoices[user.id]
            return ConversationHandler.END
        else:
            await query.message.reply_text("❌ Error saving invoice. Please try again.")
            return ConversationHandler.END
    
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel conversation."""
    user = update.effective_user
    if user.id in pending_invoices:
        del pending_invoices[user.id]
    await update.message.reply_text("❌ Operation cancelled.")
    return ConversationHandler.END

def main() -> None:
    """Start the bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.PHOTO, handle_photo)],
        states={
            ASK_STAFF_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_staff_name)],
            CONFIRM_DATE: [CallbackQueryHandler(handle_date_confirmation, pattern="^date_")],
            EDIT_DATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_date_edit)],
            SELECT_BRANCH: [CallbackQueryHandler(handle_branch_selection, pattern="^branch_")],
            ENTER_OTHER_BRANCH: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_other_branch)],
            SELECT_CATEGORY: [CallbackQueryHandler(handle_category_selection, pattern="^category_")],
            ENTER_OTHER_CATEGORY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_other_category)],
            CONFIRM_VAT: [CallbackQueryHandler(handle_vat_confirmation, pattern="^vat_")],
            CONFIRM_DATA: [CallbackQueryHandler(handle_final_confirmation, pattern="^confirm_")],
            EDIT_MENU: [CallbackQueryHandler(handle_edit_selection, pattern="^edit_")],
            EDIT_SUPPLIER: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_supplier_edit)],
            EDIT_INVOICE_NUM: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_invoice_num_edit)],
            EDIT_AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_amount_edit)],
            EDIT_ITEMS: [CallbackQueryHandler(handle_item_selection, pattern="^(items_back|item_add|item_view_|item_back_list|items_cancel)")],
            EDIT_ITEMS_ACTION: [CallbackQueryHandler(handle_item_action, pattern="^(item_delete_|itemfield_|item_cancel|item_back_list)")],
            ADD_ITEM: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_add_item)],
            EDIT_ITEM_FIELD: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_item_field_edit)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mystats", my_stats))
    application.add_handler(CommandHandler("recent", recent_invoices))
    application.add_handler(CommandHandler("myinvoices", my_invoices))
    application.add_handler(CommandHandler("edit", edit_invoice_command))
    application.add_handler(CommandHandler("delete", delete_invoice_command))
    application.add_handler(CommandHandler("confirmdelete", confirm_delete_command))
    application.add_handler(conv_handler)
    
    logger.info("Bot started successfully!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
