import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic   # pip install geopy

# -----------------------------
# Dataset (same as before)
# -----------------------------
DATA = [
    
    [1,"Goa","India","Goa",15.2993,74.1240,"beach|nightlife|water-sports","beach|party|seafood|sunset|clubs|parasailing",3500,"Nov|Dec|Jan|Feb",4.6],
    [2,"Manali","India","Manali",32.2432,77.1892,"mountains|adventure|nature","trek|snow|paragliding|river-rafting|cafe",2800,"Mar|Apr|May|Jun|Oct",4.5],
    [3,"Jaipur","India","Jaipur",26.9124,75.7873,"history|culture|city","forts|palaces|street-food|markets|heritage",3000,"Oct|Nov|Dec|Jan|Feb",4.4],
    [4,"Rishikesh","India","Rishikesh",30.0869,78.2676,"adventure|spiritual|nature","rafting|yoga|bungee|ganga|camping",2500,"Oct|Nov|Dec|Feb|Mar",4.4],
    [5,"Pondicherry","India","Puducherry",11.9139,79.8145,"beach|cafe|culture","french|beach|cycling|heritage|ashram",3000,"Oct|Nov|Dec|Jan|Feb",4.3],
    [6,"Andaman","India","Port Blair",11.6234,92.7265,"beach|diving|islands","scuba|clear-water|coral|calm|beach",5500,"Nov|Dec|Jan|Feb|Mar",4.7],
    [7,"Varanasi","India","Varanasi",25.3176,82.9739,"spiritual|history|culture","ghats|aarti|temples|saree|lassi",2200,"Nov|Dec|Jan|Feb",4.3],
    [8,"Delhi","India","New Delhi",28.6139,77.2090,"city|history|food","museums|monuments|street-food|markets",3200,"Oct|Nov|Feb|Mar",4.2],
    [9,"Gokarna","India","Gokarna",14.5479,74.3186,"beach|relaxation|backpacker","quiet|trek|beach-camping|cafes",2400,"Nov|Dec|Jan|Feb",4.4],
    [10,"Udaipur","India","Udaipur",24.5854,73.7125,"history|lakes|romantic","palaces|boat-ride|viewpoints|markets",3200,"Oct|Nov|Dec|Jan|Feb",4.5],
    [11,"Mysore","India","Mysore",12.2958,76.6394,"history|culture|city","palaces|temples|markets|gardens",2800,"Oct|Nov|Feb|Mar",4.3],
    [12,"Shimla","India","Shimla",31.1048,77.1734,"mountains|adventure|nature","trek|snow|shopping|cafe",2700,"Dec|Jan|Feb|Mar",4.4],
    [13,"Coorg","India","Madikeri",12.3375,75.8069,"hills|nature|coffee","trek|waterfalls|plantation|cafe",3000,"Oct|Nov|Dec|Jan",4.5],
    [14,"Agra","India","Agra",27.1767,78.0081,"history|culture|city","taj-mahal|forts|street-food",2500,"Oct|Nov|Feb|Mar",4.3],
    [15,"Jaisalmer","India","Jaisalmer",26.9157,70.9083,"desert|culture|history","fort|camel-safari|sand-dunes|heritage",3200,"Oct|Nov|Dec|Jan",4.4],
    [16,"Khajuraho","India","Khajuraho",24.8316,79.9196,"history|temples|culture","temples|sculptures|light-show",2800,"Oct|Nov|Feb|Mar",4.3],
    [17,"Alleppey","India","Alappuzha",9.4981,76.3388,"backwaters|relaxation|nature","houseboat|canoeing|fishing",3500,"Nov|Dec|Jan|Feb",4.6],
    [18,"Kolkata","India","Kolkata",22.5726,88.3639,"city|culture|food","museums|temples|street-food|markets",3000,"Oct|Nov|Feb|Mar",4.2],
    [19,"Darjeeling","India","Darjeeling",27.0360,88.2627,"hills|tea|nature","trek|tea-garden|cable-car|sunrise",3200,"Oct|Nov|Feb|Mar",4.5],
    [20,"Ranthambore","India","Sawai Madhopur",26.0173,76.5026,"wildlife|adventure|nature","safari|photography|fort",4000,"Oct|Nov|Feb|Mar",4.6],
    [21,"Kodaikanal","India","Kodaikanal",10.2381,77.4893,"hills|nature|romantic","trek|lake|waterfalls|boating",3000,"Oct|Nov|Dec|Jan",4.4],
    [22,"Mahabalipuram","India","Mahabalipuram",12.6131,80.1975,"history|beach|culture","temples|shore|sculptures|sunset",2800,"Oct|Nov|Dec|Jan",4.3],
    [23,"Hampi","India","Hampi",15.3350,76.4600,"history|ruins|adventure","temples|bouldering|heritage|cycling",2500,"Oct|Nov|Feb|Mar",4.5],
    [24,"Khajjiar","India","Khajjiar",32.1700,76.0500,"mountains|nature|adventure","trek|picnic|camping|lake",2800,"Jul|Aug|Sep|Oct",4.4],
    [25,"Pune","India","Pune",18.5204,73.8567,"city|culture|food","museums|cafe|shopping|temples",3000,"Oct|Nov|Feb|Mar",4.2],
    [26,"Mount Abu","India","Mount Abu",24.5930,72.7183,"hills|nature|relaxation","temples|sunset|trek|lake",3200,"Oct|Nov|Dec|Jan",4.4],
    [27,"Leh","India","Leh",34.1526,77.5770,"mountains|adventure|nature","trek|river|monastery|camping",4500,"Jun|Jul|Aug|Sep",4.6],
    [28,"Sikkim","India","Gangtok",27.3389,88.6065,"mountains|nature|culture","trek|monastery|lake|cable-car",4000,"Mar|Apr|Oct|Nov",4.5],
    [29,"Rann of Kutch","India","Bhuj",23.7336,69.6670,"desert|culture|festival","salt-lake|fair|heritage|photography",3500,"Nov|Dec|Jan",4.4],
    [30,"Ooty","India","Ooty",11.4064,76.6950,"hills|nature|romantic","botanical-garden|lake|trek|cable-car",3000,"Oct|Nov|Dec|Jan",4.5],
    [31,"Kanha","India","Kanha",22.3349,80.6110,"wildlife|nature|adventure","safari|photography|birdwatching",4000,"Oct|Nov|Feb|Mar",4.6],
    [32,"Bikaner","India","Bikaner",28.0229,73.3119,"desert|culture|history","fort|camel|palace|heritage",2800,"Oct|Nov|Feb|Mar",4.3],
    [33,"Sundarbans","India","West Bengal",21.9497,88.8183,"nature|wildlife|adventure","mangrove|safari|birdwatching|river",4500,"Oct|Nov|Dec|Jan",4.5],
    [34,"Ajmer","India","Ajmer",26.4499,74.6399,"history|spiritual|culture","dargah|temple|heritage|market",2500,"Oct|Nov|Feb|Mar",4.3],
    [35,"Pushkar","India","Pushkar",26.4909,74.5559,"spiritual|festival|culture","lake|camel|temple|fair",2800,"Oct|Nov|Feb|Mar",4.4],
    [36,"Coonoor","India","Coonoor",11.3540,76.7963,"hills|nature|tea","trek|tea-garden|lake|botanical",3000,"Oct|Nov|Dec|Jan",4.4],
    [37,"Jodhpur","India","Jodhpur",26.2389,73.0243,"history|culture|desert","fort|markets|heritage|sunset",3200,"Oct|Nov|Feb|Mar",4.5],
    [38,"Cherrapunji","India","Cherrapunji",25.2843,91.7345,"nature|waterfalls|adventure","trek|waterfalls|cave|bridge",2800,"Apr|May|Jun|Jul",4.4],
    [39,"Munnar","India","Munnar",10.0889,77.0595,"hills|tea|nature","trek|tea-garden|waterfalls|lake",3200,"Oct|Nov|Dec|Jan",4.5],
    [40,"Rameshwaram","India","Rameshwaram",9.2876,79.3129,"spiritual|beach|culture","temples|beach|heritage|fishing",3000,"Oct|Nov|Dec|Jan",4.3]
]

  

COLUMNS = ["id","name","country","city","lat","lon","categories","tags",
           "avg_cost_per_day","ideal_months","popularity"]

def budget_bucket(cost: float) -> str:
    if cost <= 2500: return "budget"
    elif cost <= 4000: return "mid"
    else: return "premium"

def get_dataframe() -> pd.DataFrame:
    return pd.DataFrame(DATA, columns=COLUMNS)

# -----------------------------
# Recommender with personalization
# -----------------------------
def build_recommender(df: pd.DataFrame):
    df = df.copy()
    df["budget_cat"] = df["avg_cost_per_day"].apply(budget_bucket)

    # TF-IDF corpus
    df["corpus"] = (
        df["tags"].fillna("") + " " +
        df["categories"].fillna("") + " " +
        df["ideal_months"].fillna("") + " " +
        df["budget_cat"].fillna("")
    )
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df["corpus"])

    def _recommend(user_prefs: dict, top_k: int = 5) -> pd.DataFrame:
        tags_part = " ".join(user_prefs.get("interests", []))
        season_part = user_prefs.get("season", "")
        pace = user_prefs.get("pace", "balanced")
        pace_hint = "slow" if pace == "relaxed" else ("fast" if pace == "fast" else "balanced")
        budget_part = user_prefs.get("budget_cat", "")
        query = f"{tags_part} {season_part} {budget_part} {pace_hint}"

        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, tfidf).ravel()
        df_scores = df.copy()
        df_scores["score"] = sims

        # --- Budget match penalty ---
        max_budget = user_prefs.get("budget", 10000)
        desired_bucket = budget_bucket(max_budget)
        df_scores["bucket_penalty"] = df_scores["budget_cat"].apply(
            lambda b: 0 if b == desired_bucket else 0.2
        )

        # --- Origin city distance factor ---
        origin_city = user_prefs.get("origin_city", None)
        if origin_city and origin_city in df["city"].values:
            origin_coords = df[df["city"] == origin_city][["lat","lon"]].values[0]
            df_scores["distance_km"] = df_scores.apply(
                lambda r: geodesic(origin_coords, (r["lat"], r["lon"])).km, axis=1
            )
            df_scores["distance_penalty"] = df_scores["distance_km"] / 2000.0  # normalize
        else:
            df_scores["distance_penalty"] = 0

        # --- Final ranking ---
        df_scores["rank_value"] = (
            df_scores["score"]
            - df_scores["bucket_penalty"]
            - df_scores["distance_penalty"]
            + 0.01*df_scores["popularity"]
            + np.random.uniform(-0.05, 0.05, len(df_scores))   # add randomness
        )

        keep_cols = ["name","country","city","avg_cost_per_day","ideal_months",
                     "categories","tags","popularity","budget_cat","rank_value"]

        return df_scores.sort_values("rank_value", ascending=False).head(top_k)[keep_cols]

    return _recommend

# -----------------------------
# Global wrapper
# -----------------------------
_df = get_dataframe()
_recommender_func = build_recommender(_df)

def recommend(user_prefs: dict, top_k: int = 5):
    return _recommender_func(user_prefs, top_k)
