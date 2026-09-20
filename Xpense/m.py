import streamlit as st
import json
import time
import google.generativeai as genai
from PIL import Image
from typing import List, Optional
import requests
from supabase import create_client, Client

# =========================================================
# PAGE CONFIGURATION & SESSION STATE
# =========================================================
st.set_page_config(page_title="Xpense", page_icon="📱", layout="wide")

st.title("📱 Xpense")
st.caption("AI-powered expense tracking via Multimodal Vision")

# Cache parsed receipts across Streamlit reruns
if "parsed_receipts" not in st.session_state:
    st.session_state.parsed_receipts = {}

# =========================================================
# API CONFIGURATION
# =========================================================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    API_KEY_READY = True
except (KeyError, FileNotFoundError):
    API_KEY_READY = False
    st.warning("⚠️ **Gemini API Key missing!** Add it to `.streamlit/secrets.toml` Governors")

# =========================================================
# SUPABASE CONFIGURATION
# =========================================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

    supabase: Client = create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )
    SUPABASE_READY = True

except (KeyError, FileNotFoundError):
    SUPABASE_READY = False
    st.warning(
        "⚠️ Supabase credentials missing! "
        "Add them to .streamlit/secrets.toml"
    )

# =========================================================
# SIDEBAR DIAGNOSTICS & TESTING
# =========================================================
with st.sidebar:
    st.subheader("⚙️ Diagnostics & Tools")
    
    # Model diagnostic tool
    if API_KEY_READY and st.button("🔍 List Available Models"):
        try:
            models = [
                m.name for m in genai.list_models() 
                if 'generateContent' in m.supported_generation_methods
            ]
            st.write("Available models for your API key:")
            st.json(models)
        except Exception as e:
            st.error(f"Error listing models: {e}")

    if st.button("🗑️ Clear Parsed Cache"):
        st.session_state.parsed_receipts = {}
        st.success("Cache cleared!")
        st.rerun()

    st.divider()
    st.subheader("🛠️ Database Tests")
    
    if SUPABASE_READY and st.button("Test REST INSERT"):
        url = f"{SUPABASE_URL}/rest/v1/receipts"
        headers = {
            "apikey": SUPABASE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        test_data = {
            "merchant": "REST Test",
            "currency": "VND",
            "purchase_date": "2026-09-20",
            "category": "📦 Other",
            "subtotal": 10000,
            "tax": 1000,
            "total": 11000
        }
        response = requests.post(url, headers=headers, json=test_data)
        st.write("Status:", response.status_code)
        st.write("Response:", response.text)

    if SUPABASE_READY and st.button("Test Supabase INSERT"):
        try:
            test_data = {
                "merchant": "Xpense Test",
                "currency": "VND",
                "purchase_date": "2026-09-20",
                "category": "📦 Other",
                "subtotal": 10000,
                "tax": 1000,
                "total": 11000
            }
            result = supabase.table("receipts").insert(test_data).execute()
            st.success("✅ INSERT worked!")
            st.write(result.data)
        except Exception as e:
            st.error(f"❌ INSERT failed: {e}")

# =========================================================
# GEMINI MULTIMODAL EXTRACTION WITH RETRY & RATE LIMITING
# =========================================================

def extract_receipt_with_vision(image: Image.Image, retries: int = 3) -> Optional[dict]:
    categories = [
        "🍔 Food", "🛒 Groceries", "🚗 Transportation", 
        "🛍️ Shopping", "🎮 Entertainment", "📚 Education", "📦 Other"
    ]
    
    prompt = f"""
    You extract structured data from receipt images. Return ONLY a valid JSON object matching this exact structure:
    {{
      "merchant": "Name of the store or restaurant or null",
      "currency": "ISO 4217 code (e.g. USD, EUR, VND, JPY) or symbol",
      "purchase_date": "Date printed on receipt in YYYY-MM-DD format or null",
      "category": "Must be exactly one of: {", ".join(categories)}",
      "items": [
        {{
          "name": "Item name",
          "price": 0.0,
          "quantity": 1
        }}
      ],
      "subtotal": 0.0,
      "tax": 0.0,
      "total": 0.0
    }}
    
    Numbers must be raw values without thousands separators. Use a period as decimal separator only when the receipt actually shows decimals.
    """
    
    model = genai.GenerativeModel('gemini-3.6-flash')
    
    for attempt in range(retries):
        try:
            response = model.generate_content(
                [prompt, image],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text.replace("```", "").strip()
                
            # Pause to stay well under free tier RPM caps
            time.sleep(1.2)
            return json.loads(raw_text)
            
        except Exception as e:
            err_msg = str(e)
            if ("429" in err_msg or "ResourceExhausted" in err_msg or "Quota" in err_msg) and attempt < retries - 1:
                wait_time = (attempt + 1) * 5
                st.warning(f"⚠️ Quota rate limit hit. Pausing {wait_time}s before retry ({attempt + 1}/{retries})...")
                time.sleep(wait_time)
            else:
                st.error(f"Vision API Error: {e}")
                return None
                
    return None

# =========================================================
# SAVE RECEIPT TO SUPABASE
# =========================================================

def save_receipt_to_database(data: dict) -> bool:
    try:
        receipt = {
            "merchant": data.get("merchant"),
            "currency": data.get("currency"),
            "purchase_date": data.get("purchase_date"),
            "category": data.get("category"),
            "subtotal": data.get("subtotal", 0),
            "tax": data.get("tax", 0),
            "total": data.get("total", 0)
        }

        receipt_response = supabase.table("receipts").insert(receipt).execute()
        
        if not receipt_response.data:
            return False

        receipt_id = receipt_response.data[0]["id"]

        items = data.get("items", [])
        if items:
            expense_items = []
            for item in items:
                expense_items.append({
                    "receipt_id": receipt_id,
                    "name": item.get("name"),
                    "price": item.get("price", 0),
                    "quantity": item.get("quantity", 1)
                })

            supabase.table("expense_items").insert(expense_items).execute()

        return True

    except Exception as e:
        st.error(f"Database error: {e}")
        return False

# =========================================================
# MAIN APP FLOW
# =========================================================

photos = st.file_uploader(
    "Upload / Take Receipt Photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if photos and API_KEY_READY:
    st.info(f"📄 {len(photos)} receipt(s) selected")
    all_data = []

    # Process each uploaded receipt
    for index, photo in enumerate(photos):
        st.divider()
        st.header(f"Receipt {index + 1} of {len(photos)}")

        try:
            image = Image.open(photo)
            image.thumbnail((1500, 1500))

            file_id = f"{photo.name}_{photo.size}"

            col_img, col_data = st.columns([1, 1.5])

            with col_img:
                st.subheader("📄 Receipt")
                st.image(image, use_container_width=True)

            with col_data:
                # Cache lookup: check session_state first before executing model call
                if file_id not in st.session_state.parsed_receipts:
                    with st.spinner(f"🤖 Analyzing receipt {index + 1}..."):
                        parsed_res = extract_receipt_with_vision(image)
                        if parsed_res:
                            st.session_state.parsed_receipts[file_id] = parsed_res

                data = st.session_state.parsed_receipts.get(file_id)

                if data:
                    all_data.append(data)
                    st.success("Receipt parsed successfully!")

                    # Main Information Metrics
                    c1, c2 = st.columns(2)
                    c1.metric("🏪 Merchant", data.get("merchant", "Unknown"))
                    c2.metric("📅 Date", data.get("purchase_date", "Unknown"))

                    c3, c4 = st.columns(2)
                    currency = data.get("currency", "")
                    c3.metric("💰 Total", f"{data.get('total', 0):,} {currency}")
                    c4.metric("🏷️ Category", data.get("category", "📦 Other"))

                    # Line Items Table
                    st.markdown("### 🛒 Line Items")
                    if data.get("items"):
                        st.dataframe(data["items"], use_container_width=True)
                    else:
                        st.info("No line items detected.")

                    # Financial Breakdown
                    with st.expander("Show Financial Breakdown (Tax/Subtotal)"):
                        st.write(f"**Subtotal:** {data.get('subtotal')} {currency}")
                        st.write(f"**Tax:** {data.get('tax')} {currency}")
                        st.write(f"**Total:** {data.get('total')} {currency}")
                else:
                    st.error(f"❌ Could not analyze receipt {index + 1}")

        except Exception as e:
            st.error(f"❌ Error opening receipt {index + 1}: {e}")

    # =====================================================
    # SUMMARY & SAVING SECTION
    # =====================================================
    if all_data:
        st.divider()
        st.header("📊 Receipt Summary")

        total_spending = sum(
            float(receipt.get("total", 0) or 0)
            for receipt in all_data
        )
        currency = all_data[0].get("currency", "")

        c1, c2 = st.columns(2)
        c1.metric("🧾 Receipts", len(all_data))
        c2.metric("💰 Total Spending", f"{total_spending:,.0f} {currency}")

        # Summary Table
        st.markdown("### 🧾 All Receipts")
        summary_data = [
            {
                "Merchant": receipt.get("merchant", "Unknown"),
                "Date": receipt.get("purchase_date", "Unknown"),
                "Category": receipt.get("category", "📦 Other"),
                "Total": receipt.get("total", 0),
                "Currency": receipt.get("currency", "")
            }
            for receipt in all_data
        ]
        st.dataframe(summary_data, use_container_width=True)

        # Save Action Button
        if st.button("💾 Save All Expenses", type="primary", use_container_width=True):
            if not SUPABASE_READY:
                st.error("Supabase is not configured. Check `.streamlit/secrets.toml`.")
            else:
                saved_count = 0
                with st.spinner("💾 Saving expenses to database..."):
                    for receipt in all_data:
                        success = save_receipt_to_database(receipt)
                        if success:
                            saved_count += 1

                if saved_count == len(all_data):
                    st.success(f"✅ Successfully saved {saved_count} receipt(s)!")
                else:
                    st.warning(f"⚠️ Saved {saved_count} out of {len(all_data)} receipts.")