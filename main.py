# app.py
import streamlit as st
import pandas as pd
from datetime import date
from recommender import get_dataframe, recommend, budget_bucket
from itinerary import generate_itinerary_llm   # AI Itinerary generator
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# --- Streamlit Config ---
st.set_page_config(page_title="AI Travel Planner", layout="wide", page_icon="✈️")

# --- Helper: safe rerun for different Streamlit versions ---
def safe_rerun():
    """Rerun the Streamlit script in a backwards/forwards-compatible way."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        # If neither exists, just raise so developer knows
        raise RuntimeError("Streamlit rerun API not available in this environment.")

# --- Global fallback origin cities (moved to top so reset_all can use it) ---
origin_cities = [
    "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad",
    "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna", "Surat", "Vadodara",
    "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi",
    "Amritsar", "Allahabad", "Coimbatore", "Madurai", "Visakhapatnam", "Vijayawada", "Guwahati",
    "Mysore", "Shimla", "Dehradun", "Ranchi", "Jabalpur"
]

# --- Helper: reset all app-specific session keys and set defaults ---
def reset_all():
    # Clear everything first
    for k in list(st.session_state.keys()):
        del st.session_state[k]

    # Put back safe defaults
    st.session_state["page"] = "landing"
    st.session_state["origin_city"] = origin_cities[0]
    st.session_state["start_date"] = date.today()
    st.session_state["trip_length"] = 5
    st.session_state["travellers"] = 2
    st.session_state["budget_usd"] = 80
    st.session_state["interests"] = []
    st.session_state["season"] = "All-year"
    st.session_state["visa_pref"] = False
    st.session_state["pace"] = "balanced"

    # Rerun to reflect changes
    safe_rerun()

# --- Page Toggle ---
if "page" not in st.session_state:
    st.session_state.page = "landing"

# =============================
# 1️⃣ Landing Page
# =============================
if st.session_state.page == "landing":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
            background-size: cover;
            background-position: center;
        }
        .landing-container {
            text-align: center;
            padding-top: 160px;
            color: white;
            text-shadow: 2px 2px 4px #000000;
        }
        .landing-title {
            font-size: 56px;
            font-weight: bold;
        }
        .landing-subtitle {
            font-size: 22px;
            margin-bottom: 40px;
        }
        .get-started-btn {
            margin-top: 24px;
        }
        </style>
        <div class="landing-container">
            <div class="landing-title">🌍 Trip Planner AI</div>
            <div class="landing-subtitle">Plan your perfect trip with AI in seconds.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚀 Get Started"):
        st.session_state.page = "main"
        safe_rerun()

# =============================
# 2️⃣ Main App (Planner)
# =============================
elif st.session_state.page == "main":

    # --- Sidebar (main app only) ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.write("Download your AI-generated itinerary after planning.")
        st.info("ℹ️ Generate an itinerary first to enable downloads.")
        st.markdown("---")
        if st.button("🧹 Clear / Reset"):
            reset_all()

    # --- Main Title ---
    st.title("✈️ AI Travel Planner — ML + GenAI + Streamlit")

    col1, col2 = st.columns(2)

    # --- Inputs ---
    with col1:
        origin_city = st.selectbox(
            "Origin city",
            origin_cities,
            index=0 if st.session_state.get("origin_city") is None else origin_cities.index(st.session_state["origin_city"]),
            key="origin_city"
        )
        start_date = st.date_input(
            "Start date",
            value=st.session_state.get("start_date", date.today()),
            key="start_date"
        )
        trip_length = st.number_input(
            "Trip length (days)",
            min_value=1,
            value=st.session_state.get("trip_length", 5),
            step=1,
            key="trip_length"
        )

    with col2:
        travellers = st.number_input(
            "Travellers",
            min_value=1,
            value=st.session_state.get("travellers", 2),
            step=1,
            key="travellers"
        )
        
        budget_usd = st.slider(
            "Budget per person per day (USD)",
            20, 500,
            value=st.session_state.get("budget_usd", 80),
            key="budget_usd"
        )
        usd_to_inr = 83
        budget_inr = budget_usd * usd_to_inr
        st.write(f"💰 Budget per person per day: ₹{budget_inr:.0f} INR")

        interests = st.multiselect(
            "Interests",
            ["food", "culture", "shopping", "adventure", "nature"],
            default=st.session_state.get("interests", []),
            key="interests"
        )
        season = st.selectbox(
            "Season / Window",
            ["All-year", "Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"],
            index=["All-year", "Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"].index(
                st.session_state.get("season", "All-year")
            ),
            key="season"
        )
        visa_pref = st.checkbox("Prefer visa-on-arrival / e-visa", value=st.session_state.get("visa_pref", False), key="visa_pref")
        pace = st.selectbox(
            "Pace",
            ["relaxed", "balanced", "fast"],
            index=["relaxed", "balanced", "fast"].index(st.session_state.get("pace", "balanced")) ,
            key="pace"
        )

    # --- Recommendation Section ---
    st.subheader("🔮 Recommended destinations")

    if st.button("Find destinations"):
        user_prefs = {
            "budget": budget_inr,
            "season": season,
            "interests": interests,
            "pace": pace,
            "days": trip_length
        }
        recs = recommend(user_prefs, top_k=5)
        st.session_state["recs"] = recs

    # --- Quick links helper ---
    def generate_links(city, country):
        city_query = f"{city},{country}".replace(" ", "+")
        links = {
            "Maps": f"https://www.google.com/maps/search/{city_query}",
            "Hotels": f"https://www.booking.com/searchresults.html?ss={city_query}",
            "Flights": f"https://www.google.com/flights?hl=en#flt={city_query}"
        }
        return links

    # --- Display recommendations ---
    if "recs" in st.session_state and not st.session_state["recs"].empty:
        recs = st.session_state["recs"]
        st.dataframe(recs)

        st.subheader("🗺️ Build itinerary for:")
        recs["display"] = recs["name"] + " (" + recs["country"] + ")"
        selected_display = st.selectbox("Select destination", recs["display"])
        selected_row = recs[recs["display"] == selected_display].iloc[0]

        # Quick Links buttons
        links = generate_links(selected_row.get("city", ""), selected_row["country"])
        col_maps, col_hotels, col_flights = st.columns(3)
        with col_maps:
            st.markdown(f"[🗺️ Maps]({links['Maps']})", unsafe_allow_html=True)
        with col_hotels:
            st.markdown(f"[🏨 Hotels]({links['Hotels']})", unsafe_allow_html=True)
        with col_flights:
            st.markdown(f"[✈️ Flights]({links['Flights']})", unsafe_allow_html=True)

        if st.button("Generate itinerary ✨"):
            out = generate_itinerary_llm(
                destination=selected_row["name"],
                country=selected_row["country"],
                days=trip_length,
                start_date=start_date,
                interests=interests,
                budget_cat=f"₹{budget_inr:.0f}/day",
                travellers=travellers,
                pace=pace
            )
            st.session_state["itinerary"] = out
    
    # --- Display itinerary & downloads ---
    if "itinerary" in st.session_state:
        st.markdown(st.session_state["itinerary"])

        # --- Download buttons (Main Panel) ---
        st.subheader("📥 Download your itinerary")

        itinerary_text = st.session_state["itinerary"]

        # TXT download
        st.download_button(
            label="⬇️ Download Itinerary (TXT)",
            data=itinerary_text,
            file_name="itinerary.txt",
            mime="text/plain"
        )

        # PDF download
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = [Paragraph(line, styles["Normal"]) for line in itinerary_text.split("\n")]
        doc.build(story)
        pdf_bytes = buffer.getvalue()

        st.download_button(
            label="📄 Download Itinerary (PDF)",
            data=pdf_bytes,
            file_name="itinerary.pdf",
            mime="application/pdf"
        )


