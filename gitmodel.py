import os
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow and other logs
import time
import re
import math
import pandas as pd
import numpy as np
from fractions import Fraction
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from jinja2 import Template

def generate_race_html(race_id, race_name, df):
    """
    Generate an HTML file for a race, with colored horse names based on criteria:
    - Gold background if 'Value' indicator present.
    - Blue background if real odds >= 10 and model odds < real odds.
    """
    html_template = Template(r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ race_name }}</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5; }
    h1 { text-align: center; color: #333; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 12px; text-align: center; border-bottom: 1px solid #ddd; }
    th { background-color: #333; color: #fff; }
    tr:nth-child(even) { background-color: #f2f2f2; }
    .value-horse { background-color: gold !important; }
    .undervalued { background-color: lightblue !important; }
  </style>
</head>
<body>
  <h1>{{ race_name }}</h1>
  <table>
    <thead>
      <tr>
        <th>Horse Name</th><th>Real Odds</th><th>Model Odds</th><th>Score</th>
        <th>Going</th><th>Distance</th><th>Class</th><th>Form</th><th>Value</th>
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row['Horse Name'] }}</td>
        <td>{{ row['Real Odds'] }}</td>
        <td>{{ row['Model Odds'] }}</td>
        <td>{{ row['Score'] }}</td>
        <td>{{ row['Going'] }}</td>
        <td>{{ row['Distance'] }}</td>
        <td>{{ row['Class'] }}</td>
        <td>{{ row['Form'] }}</td>
        <td>{{ row['Value'] }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  <script>
    document.querySelectorAll('table tbody tr').forEach(row => {
      const cells = row.querySelectorAll('td');
      const horseCell = cells[0];
      const realOdds = parseFloat(cells[1].textContent) || 0;
      const modelOdds = parseFloat(cells[2].textContent) || 0;
      const valueText = cells[8].textContent;

      if (valueText.includes('✔') || valueText.includes('💰')) {
        horseCell.classList.add('value-horse');
      } else if (realOdds >= 10 && modelOdds < realOdds) {
        horseCell.classList.add('undervalued');
      }
    });
  </script>
</body>
</html>
    """)
    rows = df.to_dict(orient='records')
    rendered = html_template.render(race_name=race_name, rows=rows)
    filename = f"{race_id}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(rendered)
    return filename


import numpy as np

def softmax(x):
    """Apply softmax to a list of scores for better odds differentiation."""
    x = np.array(x)
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()

# --- Add adjust_weight helper ---
def adjust_weight(weights, key, multiplier, cap=1.2):
    """Multiply weights[key] by multiplier, but cap the increase to 'cap' times the original value."""
    weights[key] = min(weights[key] * multiplier, weights[key] * cap)



##############################
# Scraping Functions
##############################
def extract_race_type_from_name(race_name):
    name = race_name.lower()

    # PRIORITY: Combined or specific subtypes
    if "juvenile handicap hurdle" in name:
        return "Juvenile Handicap Hurdle"
    elif "juvenile hurdle" in name:
        return "Juvenile Hurdle"
    elif "juvenile chase" in name:
        return "Juvenile Chase"
    elif "veterans' handicap chase" in name or "veterans handicap chase" in name:
        return "Veterans Handicap Chase"
    elif "veterans' chase" in name or "veterans chase" in name:
        return "Veterans Chase"
    elif "novice handicap hurdle" in name:
        return "Novice Handicap Hurdle"
    elif "novice handicap chase" in name:
        return "Novice Handicap Chase"
    elif "novice hurdle" in name or "novices' hurdle" in name:
        return "Novice Hurdle"
    elif "novice chase" in name or "novices' chase" in name:
        return "Novice Chase"
    elif "national hunt flat" in name or "nh flat" in name or "bumper" in name:
        return "Bumper"
    elif "handicap hurdle" in name:
        return "Handicap Hurdle"
    elif "handicap chase" in name:
        return "Handicap Chase"
    elif "group 1" in name:
        return "Group 1"
    elif "group 2" in name:
        return "Group 2"
    elif "group 3" in name:
        return "Group 3"
    elif "listed" in name:
        return "Listed"
    elif "conditions stakes" in name or "conditions race" in name:
        return "Conditions"
    elif "maiden handicap" in name:
        return "Maiden Handicap"
    elif "classified stakes" in name:
        return "Classified Stakes"
    elif "maiden" in name:
        return "Maiden"
    elif "selling" in name or "seller" in name:
        return "Selling"
    elif "claimer" in name or "claiming" in name:
        return "Claiming"
    elif "apprentice" in name or "lady riders" in name or "gentleman riders" in name:
        return "Apprentice"
    elif "handicap" in name:
        return "Handicap"
    elif "hurdle" in name:
        return "Hurdle"
    elif "chase" in name or "steeplechase" in name:
        return "Chase"
    else:
        return "Other"




def fix_weight(weight_str):
    if "-" in weight_str:
        parts = weight_str.split("-")
        if len(parts) == 2:
            first, second = parts[0].strip(), parts[1].strip()
            month_map = {
                "Jan": "1", "Feb": "2", "Mar": "3", "Apr": "4", "May": "5",
                "Jun": "6", "Jul": "7", "Aug": "8", "Sep": "9", "Oct": "10",
                "Nov": "11", "Dec": "12"
            }
            if second[:3] in month_map:
                try:
                    fixed_first = str(int(first))
                except:
                    fixed_first = first
                fixed_second = month_map[second[:3]]
                return f"{fixed_first}-{fixed_second}"
    return weight_str

def parse_weight_to_lbs(weight_str):
    try:
        stones, pounds = weight_str.split("-")
        return int(stones) * 14 + int(pounds)
    except:
        return np.nan

def extract_distance(text):
    """
    Extracts distances like '1m', '7f', '1m 2f', '6f 110y', etc.
    Ignores age-related text like '3YO only'.
    """
    # Accept miles, furlongs, and yards patterns only if they follow race type/conditions.
    # Avoids misreading age references like '3YO only'
    match = re.search(r"(\d+m(?: \d+f)?(?: \d+y)?)|(\d+f(?: \d+y)?)", text)
    if match:
        return match.group(0).strip()
    return "Unknown"

def fetch_race_card_data(url):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    import os
    import pandas as pd
    from datetime import datetime

    service = Service(ChromeDriverManager().install(), service_log_path=os.devnull)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--log-level=3")
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(5)

    # ✅ Race Metadata
    try:
        time_location_text = driver.find_element(By.CSS_SELECTOR, "p.CourseListingHeader__StyledMainTitle-sc-af53af6-5").text
        parts = time_location_text.split()
        race_time = parts[0].replace(":", "-")
        race_location = " ".join(parts[1:])
    except:
        race_time = "Unknown"
        race_location = "Unknown"


    try:
        date_text = driver.find_element(By.CSS_SELECTOR, "p.CourseListingHeader__StyledMainSubTitle-sc-af53af6-7").text
        race_date = datetime.strptime(date_text, "%A %d %B %Y").strftime("%d/%m/%Y")
    except:
        race_date = "Unknown"

    try:
        race_name = driver.find_element(By.CSS_SELECTOR, "h1[data-test-id='racecard-race-name']").text
    except:
        race_name = "Unknown"

    try:
        meta_text = driver.find_element(By.CSS_SELECTOR, "li.RacingRacecardSummary__StyledAdditionalInfo-sc-ff7de2c2-3").text
        parts = [p.strip() for p in meta_text.split("|")]

        # Distance part (e.g., '1m 110y')
        # Use improved extraction to avoid picking up "3YO only"
        distance = "Unknown"
        for p in parts:
            # Only consider lines that look like a distance
            if any(unit in p for unit in ["f", "m", "y"]):
                d = extract_distance(p)
                if d != "Unknown":
                    distance = d
                    break

        # Going logic – includes Turf, AW and US surfaces
        raw_going = next((
            p for p in parts if any(g in p.lower() for g in [
                "good", "firm", "soft", "heavy", "yielding",             # UK Turf
                "standard", "standard to slow", "standard to fast",      # All-Weather
                "standard/slow", "standard/fast", "slow",                # More AW
                "fast", "sloppy", "muddy", "wet", "dry"                  # US
            ])
        ), "Unknown")
        going = raw_going.split("(")[0].strip().title()  # Title-case it for formatting



        # Class (may not exist in US races)
        race_class = next((p for p in parts if "Class" in p or "Group" in p or "Grade" in p), "Unknown")

        # Combined race type metadata
        race_type_data = f"{distance} | {going} | {race_class}"

    except Exception as e:
        print(f"⚠️ Failed to parse race type meta: {e}")
        race_type_data = "Unknown"





    csv_filename = f"{race_time}_{race_location}.csv" if race_time != "Unknown" else "Unknown_Race.csv"

    # ✅ Runner Data
    runners = driver.find_elements(By.CLASS_NAME, "Runner__StyledRunnerContainer-sc-c8a39dcf-0")
    race_data = []



    # Parse race distance in yards from race_type_data
    def get_todays_distance(race_type_data):
        import re
        if not race_type_data or not isinstance(race_type_data, str):
            return None
        race_type_data = race_type_data.lower().strip()
        miles = furlongs = yards = 0
        mile_match = re.search(r"(\d+)\s*m", race_type_data)
        furlong_match = re.search(r"(\d+)\s*f", race_type_data)
        yard_match = re.search(r"(\d+)\s*y", race_type_data)
        if not furlong_match:
            furlong_match = re.search(r"(\d+)f", race_type_data)
        if not yard_match:
            yard_match = re.search(r"(\d+)y", race_type_data)
        if mile_match:
            miles = int(mile_match.group(1))
        if furlong_match:
            furlongs = int(furlong_match.group(1))
        if yard_match:
            yards = int(yard_match.group(1))
        total_yards = (miles * 1760) + (furlongs * 220) + yards
        return total_yards if total_yards > 0 else None

    race_distance_yards = get_todays_distance(race_type_data) or "Unknown"

    for runner in runners:
        try:
            horse_name = runner.find_element(By.CSS_SELECTOR, "a[data-test-id='horse-name-link']").text
        except:
            horse_name = "Unknown"

        try:
            odds = runner.find_element(By.CLASS_NAME, "BetLink__BetLinkStyle-sc-7392938a-0").text
            odds = f"'{odds}"
        except:
            odds = "No odds available"

        try:
            headgear = runner.find_element(By.CSS_SELECTOR, "sup[data-test-id='headgear']").text
        except:
            headgear = "None"

        try:
            last_ran = runner.find_element(By.CSS_SELECTOR, "sup[data-test-id='last-ran']").text
        except:
            last_ran = "Unknown"

        try:
            saddle_cloth = runner.find_element(By.CLASS_NAME, "SaddleAndStall__StyledSaddleClothNo-sc-2df3fa22-1").text
        except:
            saddle_cloth = "Unknown"

        try:
            stall_number = runner.find_element(By.CLASS_NAME, "SaddleAndStall__StyledStallNo-sc-2df3fa22-2").text
        except:
            stall_number = "Unknown"

        sub_info = runner.find_elements(By.CLASS_NAME, "Runner__StyledSubInfoLink-sc-c8a39dcf-16")
        jockey = sub_info[0].text if len(sub_info) > 0 else "Unknown"
        trainer = sub_info[1].text if len(sub_info) > 1 else "Unknown"

        try:
            horse_info = runner.find_element(By.CLASS_NAME, "Runner__StyledSubInfo-sc-c8a39dcf-4").text
            parts = [p.strip() for p in horse_info.split("|")]
            age = weight = official_rating = "Unknown"
            for part in parts:
                if "Age:" in part:
                    age = part.replace("Age:", "").strip()
                elif "Weight:" in part:
                    weight = part.replace("Weight:", "").strip()
                elif "OR:" in part:
                    official_rating = part.replace("OR:", "").strip()
        except:
            age, weight, official_rating = "Unknown", "Unknown", "Unknown"

        try:
            form_string = runner.find_element(By.CLASS_NAME, "Runner__StyledFormButton-sc-c8a39dcf-3").text.replace("Form:", "").strip()
        except:
            form_string = "No form available"

        try:
            comments = runner.find_element(By.CSS_SELECTOR, "div[data-test-id='commentary']").text
        except:
            comments = "No comments available"

        # ✅ Expand past form table and extract
        past_form_list = []
        past_race_history = "Unknown"
        try:
            expand_button = runner.find_element(By.CLASS_NAME, "Runner__StyledFormButton-sc-c8a39dcf-3")
            driver.execute_script("arguments[0].scrollIntoView(true);", expand_button)
            driver.execute_script("arguments[0].click();", expand_button)
            time.sleep(1.5)

            form_wrapper = runner.find_element(By.CLASS_NAME, "FormTable__FormTableWrapper-sc-b08621f3-0")
            form_table = form_wrapper.find_element(By.TAG_NAME, "table")
            rows = form_table.find_elements(By.TAG_NAME, "tr")[1:]

            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 7:
                    date = cols[0].text.strip()
                    course = cols[1].text.strip()
                    race_class = cols[2].text.strip() or "N/A"
                    distance = cols[3].text.strip()
                    going = cols[4].text.strip()
                    orating = cols[5].text.strip()
                    position = cols[6].text.strip()
                    past_form_list.append(
                        f"Date: {date} | Course: {course} | Class: {race_class} | Distance: {distance} | Going: {going} | OR: {orating} | Position: {position}"
                    )

            if past_form_list:
                past_race_history = " |||| ".join(past_form_list)
        except:
            print(f"⚠️ '{horse_name}' has no past form available.")

        if not isinstance(past_race_history, str) or past_race_history.strip() == "":
            past_race_history = "Unknown"

        print("🧪 Entry length:", len([
           race_date, race_time.replace("-", ":"), race_location, race_name, race_type_data, race_distance_yards,
            horse_name, headgear, last_ran, saddle_cloth, stall_number, jockey, trainer, age, weight,
            official_rating, form_string, comments, odds, past_race_history
        ]))


        race_data.append([
            race_date, race_time.replace("-", ":"), race_location, race_name, race_type_data, race_distance_yards,
            horse_name, headgear, last_ran, saddle_cloth, stall_number, jockey, trainer, age, weight,
            official_rating, form_string, comments, odds, past_race_history
    ])



    columns = [
    "Race Date", "Race Time", "Race Location", "Race Name", "Race Type Data", "Race Distance Yards",
    "Horse Name", "Headgear", "Last Ran (Days)", "Saddle Cloth", "Stall", "Jockey", "Trainer", "Age",
    "Weight", "Official Rating", "Recent Form", "Comments", "Odds", "Past Race History"
]




    df = pd.DataFrame(race_data, columns=columns)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    print()  # Blank line 1
    print()  # Blank line 2
    print("=====================================")
    print(f"Race card data successfully saved as '{csv_filename}'.")
    return csv_filename





##############################
# Modelling Helper Functions
##############################
def assess_class_change(row, today_class):
    """
    Determines if a horse is moving up, down or staying in the same class,
    and how they've performed historically at different class levels.
    """
    try:
        past_classes = eval(row.get("Past Race Classes", "[]"))  # List like [4, 4, 5]
        if not past_classes:
            return "unknown", 0, 0

        same_or_higher = [cls for cls in past_classes if cls <= today_class]
        lower = [cls for cls in past_classes if cls > today_class]

        direction = "same"
        if all(cls > today_class for cls in past_classes):
            direction = "up"
        elif all(cls < today_class for cls in past_classes):
            direction = "down"

        experience_score = len(same_or_higher)
        inexperience_penalty = len(lower)

        return direction, experience_score, inexperience_penalty
    except Exception:
        return "unknown", 0, 0





def parse_headgear_factor(headgear, comments):
    if not isinstance(headgear, str):
        return 0.0
    h = headgear.lower().strip()
    if h == "none" or h == "":
        return 0.0
    bonus = 0.0
    if "blinkers" in h:
        bonus += 0.10
    if "visor" in h:
        bonus += 0.05
    if "hood" in h:
        bonus += 0.07
    if bonus == 0.0:
        bonus = 0.03
    comments_lower = comments.lower()
    if "first-time" in comments_lower:
        if "blinkers" in comments_lower and "blinkers" in h:
            bonus += 0.05
        elif "visor" in comments_lower and "visor" in h:
            bonus += 0.03
        elif "hood" in comments_lower and "hood" in h:
            bonus += 0.04
    return bonus

def age_factor(age_str, optimal=7, std=3):
    try:
        age = float(age_str)
        return math.exp(-((age - optimal) ** 2) / (2 * (std ** 2)))
    except:
        return 0.5

def parse_distance(distance_str):
    miles = 0
    furlongs = 0
    yards = 0
    match = re.search(r"(\d+)m", distance_str)
    if match:
        miles = int(match.group(1))
    match = re.search(r"(\d+)f", distance_str)
    if match:
        furlongs = int(match.group(1))
    match = re.search(r"(\d+)y", distance_str)
    if match:
        yards = int(match.group(1))
    total_yards = miles * 1760 + furlongs * 220 + yards
    return total_yards

def get_todays_distance(race_type_data):
    """
    Parses race_type_data like '1m 5f 192y', '6f213y', '2m87y' etc. and returns total distance in yards.
    """
    import re

    if not race_type_data or not isinstance(race_type_data, str):
        return None

    # Lowercase and strip
    race_type_data = race_type_data.lower().strip()

    # Initialize
    miles = furlongs = yards = 0

    # Match patterns
    mile_match = re.search(r"(\d+)\s*m", race_type_data)
    furlong_match = re.search(r"(\d+)\s*f", race_type_data)
    yard_match = re.search(r"(\d+)\s*y", race_type_data)

    # Also support compact formats like '6f213y'
    if not furlong_match:
        furlong_match = re.search(r"(\d+)f", race_type_data)
    if not yard_match:
        yard_match = re.search(r"(\d+)y", race_type_data)

    # Convert if present
    if mile_match:
        miles = int(mile_match.group(1))
    if furlong_match:
        furlongs = int(furlong_match.group(1))
    if yard_match:
        yards = int(yard_match.group(1))

    total_yards = (miles * 1760) + (furlongs * 220) + yards
    return total_yards if total_yards > 0 else None


def get_todays_going(race_type_data):
    """
    Extracts going from race_type_data string like '1m 2f | Good To Firm | Class 6'
    """
    if not isinstance(race_type_data, str):
        return "Unknown"

    parts = [p.strip() for p in race_type_data.split("|")]
    for part in parts:
        part_lower = part.lower()
        if any(keyword in part_lower for keyword in [
            "good", "firm", "soft", "heavy", "yielding",
            "standard", "standard to slow", "standard to fast",
            "slow", "fast", "muddy", "sloppy", "wet"
        ]):
            return part.title()  # Ensure 'Good To Firm' formatting
    return "Unknown"


def parse_class_from_race_type(race_type_data):
    match = re.search(r"Class (\d+)", race_type_data)
    if match:
        return int(match.group(1))
    return None

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%d/%m/%y")
    except:
        return None

def last_ran_factor(last_ran_str, distance_yards):
    try:
        last_ran = float(last_ran_str)
    except:
        return 1.0
    if distance_yards <= 0:
        return 1.0
    baseline = 10 * (distance_yards / 5000)
    if last_ran >= baseline:
        return math.exp(-(last_ran - baseline) / baseline)
    else:
        return 1 + ((baseline - last_ran) / baseline) * 0.5

def weight_factor(horse_weight_str, avg_weight):
    weight_lbs = parse_weight_to_lbs(horse_weight_str)
    if weight_lbs and avg_weight > 0:
        return math.pow(avg_weight / weight_lbs, 0.5)
    return 0.5

def recent_form_factor(form_str):
    if not isinstance(form_str, str) or form_str.lower().strip() in ["", "no form available"]:
        return 0.5
    tokens = re.split(r'[\s,-]+', form_str.strip())
    scores = []
    for i, token in enumerate(reversed(tokens)):
        if token.isdigit():
            pos = int(token)
            if pos == 1:
                score = 1.0
            elif pos <= 3:
                score = 0.5
            elif pos <= 8:
                score = 0.25
            else:
                score = -0.5
        elif token.upper() in ["F", "U", "P"]:
            score = -2.0
        else:
            continue
        weight = 1 / (i + 1)
        scores.append(score * weight)
    if scores:
        total_score = sum(scores)
        total_weight = sum(1 / (i + 1) for i in range(len(scores)))
        weighted_avg = total_score / total_weight if total_weight > 0 else 0
        return 1 + 0.2 * weighted_avg
    else:
        return 0.5

def comments_sentiment_factor(comments):
    try:
        from textblob import TextBlob
        blob = TextBlob(comments)
        polarity = blob.sentiment.polarity
        return 1 + polarity
    except ImportError:
        score = simple_sentiment(comments)
        return 1 + 0.1 * score
    except:
        return 1.0

def simple_sentiment(comments):
    if not isinstance(comments, str):
        return 0
    text = comments.lower()
    score = 0
    positive_words = [
        "win", "favour", "good", "strong", "excellent", "headway", "ran on", "stayed on", "kept on", 
        "quickened", "challenge", "led", "cosily", "comfortable", "impressive", "dominant", "fluent", 
        "powerful", "smart", "game", "brave", "battled", "eased", "burst", "cruised", "relished", 
        "promising", "progressed", "nearest finish"
    ]
    negative_words = [
        "loss", "poor", "unlucky", "weak", "weakened", "no impression", "not quite pace", "hung left", 
        "same pace", "regressed", "no extra", "faded", "struggled", "outpaced", "laboured", "tailed off", 
        "never involved", "disappointing", "found little", "flat", "dropped away", "beaten", "hampered", 
        "awkward", "slow"
    ]
    for word in positive_words:
        if word in text:
            score += 1.5
    for word in negative_words:
        if word in text:
            score -= 1.5
    return score

def course_factor(past_history, race_location, race_date):
    if not isinstance(past_history, str) or not past_history.strip():
        return 0.5
    today = parse_date(race_date) or datetime.now()
    races = re.split(r"\|\|\|\|", past_history)
    scores = []
    for race in races:
        date_match = re.search(r"Date:\s*([^|]+)", race)
        course_match = re.search(r"Course:\s*([^|]+)", race)
        pos_match = re.search(r"Position:\s*(\d+)\s*/\s*(\d+)", race)
        if date_match and course_match and pos_match:
            past_date_str = date_match.group(1).strip()
            past_course = course_match.group(1).strip().lower()
            pos = float(pos_match.group(1))
            total = float(pos_match.group(2))
            past_date = parse_date(past_date_str)
            if past_course == race_location.lower() and past_date:
                days_ago = (today - past_date).days
                decay = math.exp(-days_ago / 180)
                if total > 1:
                    score = (1 - (pos - 1) / (total - 1)) * decay
                else:
                    score = (1 if pos == 1 else 0) * decay
                scores.append(score)
    if scores:
        return sum(scores) / len(scores)
    return 0.5

def parse_fractional_odds(odds_str):
    try:
        odds_str = odds_str.strip().strip("'")
        num, denom = odds_str.split("/")
        return float(num) / float(denom)
    except:
        return np.nan

def parse_fractional_odds_to_decimal(odds_str):
    try:
        odds_str = odds_str.strip().strip("'")
        num, denom = odds_str.split("/")
        fractional_odds = float(num) / float(denom)
        return fractional_odds + 1  # Convert to decimal odds
    except:
        return np.nan

def parse_past_performance(past_history, race_date):
    if not isinstance(past_history, str) or not past_history.strip():
        return 0.5
    today = parse_date(race_date) or datetime.now()
    races = re.split(r"\|\|\|\|", past_history)
    scores = []
    for race in races:
        date_match = re.search(r"Date:\s*([^|]+)", race)
        pos_match = re.search(r"Position:\s*(\d+)\s*/\s*(\d+)", race)
        if date_match and pos_match:
            past_date_str = date_match.group(1).strip()
            pos = float(pos_match.group(1))
            total = float(pos_match.group(2))
            past_date = parse_date(past_date_str)
            if past_date:
                days_ago = (today - past_date).days
                decay = math.exp(-days_ago / 180)
                if total > 1:
                    score = (1 - (pos - 1) / (total - 1)) * decay
                else:
                    score = (1 if pos == 1 else 0) * decay
                scores.append(score)
    if scores:
        return sum(scores) / len(scores)
    return 0.5

def parse_similar_performance(past_history, todays_course, todays_distance, todays_going, todays_class, race_date):
    if not past_history or not todays_course or not todays_distance or not todays_going or not todays_class:
        return 0.5
    today = parse_date(race_date) or datetime.now()
    races = re.split(r"\|\|\|\|", past_history)
    scores = []
    for race in races:
        date_match = re.search(r"Date:\s*([^|]+)", race)
        course_match = re.search(r"Course:\s*([^|]+)", race)
        distance_match = re.search(r"Distance:\s*([^|]+)", race)
        going_match = re.search(r"Going:\s*([^|]+)", race)
        class_match = re.search(r"Class:\s*(\d+)", race)
        pos_match = re.search(r"Position:\s*(\d+)\s*/\s*(\d+)", race)
        if date_match and course_match and distance_match and going_match and class_match and pos_match:
            past_date_str = date_match.group(1).strip()
            past_course = course_match.group(1).strip().lower()
            past_distance_str = distance_match.group(1).strip()
            past_going = going_match.group(1).strip().lower()
            past_class = int(class_match.group(1))
            pos = float(pos_match.group(1))
            total = float(pos_match.group(2))
            past_date = parse_date(past_date_str)
            past_distance = parse_distance(past_distance_str)
            if past_date and past_course == todays_course.lower() and abs(past_distance - todays_distance) / todays_distance <= 0.1 and past_going == todays_going and past_class == todays_class:
                days_ago = (today - past_date).days
                decay = math.exp(-days_ago / 180)
                if total > 1:
                    score = (1 - (pos - 1) / (total - 1)) * decay
                else:
                    score = (1 if pos == 1 else 0) * decay
                scores.append(score)
    if scores:
        return sum(scores) / len(scores)
    return 0.5

def going_suitability(past_history, todays_going, race_date):
    if not past_history or not todays_going:
        return None  # Changed from 0.5 to None

    today = parse_date(race_date) or datetime.now()
    races = re.split(r"\|\|\|\|", past_history)
    scores = []

    def normalize_going(text):
        text = text.lower().strip()
        # Collapse common phrases
        if "good to firm" in text:
            return "good to firm"
        elif "good to soft" in text:
            return "good to soft"
        elif "firm" in text:
            return "firm"
        elif "soft" in text:
            return "soft"
        elif "heavy" in text:
            return "heavy"
        elif "yielding" in text:
            return "yielding"
        elif "good" in text:
            return "good"
        elif "standard to slow" in text:
            return "standard to slow"
        elif "standard to fast" in text:
            return "standard to fast"
        elif "standard" in text:
            return "standard"
        return "unknown"

    def normalize_group(going):
        """
        Groups going descriptions into broader categories for fuzzy matching.
        """
        g = going.lower().strip()
        if g in ["good", "good to firm", "good to soft"]:
            return "good"
        elif g in ["firm", "standard to fast"]:
            return "firm"
        elif g in ["soft", "yielding"]:
            return "soft"
        elif g == "heavy":
            return "heavy"
        elif g in ["standard", "standard to slow"]:
            return "standard"
        else:
            return "unknown"

    target = normalize_going(todays_going)
    target_group = normalize_group(target)

    for race in races:
        date_match = re.search(r"Date:\s*([^|]+)", race)
        going_match = re.search(r"Going:\s*([^|]+)", race)
        pos_match = re.search(r"Position:\s*(\d+)\s*/\s*(\d+)", race)

        if date_match and going_match and pos_match:
            past_date_str = date_match.group(1).strip()
            past_going_raw = going_match.group(1).strip()
            past_going = normalize_going(past_going_raw)
            past_group = normalize_group(past_going)
            pos = float(pos_match.group(1))
            total = float(pos_match.group(2))
            past_date = parse_date(past_date_str)

            # --- Fuzzy group match instead of exact match ---
            if past_date and past_group != "unknown" and past_group == target_group:
                days_ago = (today - past_date).days
                decay = math.exp(-days_ago / 180)
                if total > 1:
                    score = (1 - (pos - 1) / (total - 1)) * decay
                else:
                    score = (1 if pos == 1 else 0) * decay
                scores.append(score)

    if scores:
        return sum(scores) / len(scores)
    return None  # Changed from 0.5 to None

def distance_suitability(past_history, todays_distance, race_date):
    if not past_history or not todays_distance:
        return None  # Changed from 0.5 to None
    today = parse_date(race_date) or datetime.now()
    races = re.split(r"\|\|\|\|", past_history)
    scores = []
    for race in races:
        date_match = re.search(r"Date:\s*([^|]+)", race)
        # Improved regex for distance parsing
        distance_match = re.search(r"Distance:\s*([\d\s]*m)?\s*([\d\s]*f)?\s*([\d\s]*y)?", race, re.IGNORECASE)
        pos_match = re.search(r"Position:\s*(\d+)\s*/\s*(\d+)", race)
        if date_match and distance_match and pos_match:
            past_date_str = date_match.group(1).strip()
            # Parse miles, furlongs, yards robustly
            miles = int(distance_match.group(1).strip().replace("m", "")) if distance_match.group(1) else 0
            furlongs = int(distance_match.group(2).strip().replace("f", "")) if distance_match.group(2) else 0
            yards = int(distance_match.group(3).strip().replace("y", "")) if distance_match.group(3) else 0
            past_distance = miles * 1760 + furlongs * 220 + yards
            pos = float(pos_match.group(1))
            total = float(pos_match.group(2))
            past_date = parse_date(past_date_str)
            if past_date and past_distance and abs(past_distance - todays_distance) / todays_distance <= 0.1:
                days_ago = (today - past_date).days
                decay = math.exp(-days_ago / 180)
                if total > 1:
                    score = (1 - (pos - 1) / (total - 1)) * decay
                else:
                    score = (1 if pos == 1 else 0) * decay
                scores.append(score)
    if scores:
        return sum(scores) / len(scores)
    return None  # Changed from 0.5 to None

def class_factor(past_history, todays_class, race_date):
    if not past_history or not todays_class:
        return None  # Changed from 0.5 to None
    today = parse_date(race_date) or datetime.now()
    races = re.split(r"\|\|\|\|", past_history)
    same_class_scores = []
    higher_class_scores = []
    lower_class_scores = []

    for race in races:
        date_match = re.search(r"Date:\s*([^|]+)", race)
        class_match = re.search(r"Class:\s*(\d+)", race)
        pos_match = re.search(r"Position:\s*(\d+)\s*/\s*(\d+)", race)
        if date_match and class_match and pos_match:
            past_date_str = date_match.group(1).strip()
            past_class = int(class_match.group(1))
            pos = float(pos_match.group(1))
            total = float(pos_match.group(2))
            past_date = parse_date(past_date_str)
            if past_date and total > 1:
                days_ago = (today - past_date).days
                decay = math.exp(-days_ago / 180)
                score = (1 - (pos - 1) / (total - 1)) * decay  # Normalized position with recency decay
                if past_class == todays_class:
                    same_class_scores.append(score)
                elif past_class < todays_class:  # Higher class (lower number)
                    higher_class_scores.append(score)
                elif past_class > todays_class:  # Lower class (higher number)
                    lower_class_scores.append(score)

    base_score = sum(same_class_scores) / len(same_class_scores) if same_class_scores else None

    adjustment = 0
    if higher_class_scores:  # Dropping down in class
        avg_higher = sum(higher_class_scores) / len(higher_class_scores)
        adjustment += 0.5 * avg_higher  # Bonus for proven success in tougher races
    if lower_class_scores:  # Moving up in class
        avg_lower = sum(lower_class_scores) / len(lower_class_scores)
        adjustment -= 0.25 * (1 - avg_lower)  # Penalty if struggled in easier races, reduced if did well

    if base_score is None:
        return None  # or -1 if you're using sentinel values
    return max(0.0, min(1.0, base_score + adjustment))

# --- Updated standardized score_to_emoji helper ---
def score_to_emoji(score):
    if score is None or score == -1:
        return "❓"
    elif score >= 0.75:
        return "🔥"
    elif score >= 0.50:
        return "🟢"
    elif score >= 0.40:
        return "🟡"
    elif score >= 0.15:
        return "🟠"
    else:
        return "🔴"

# --- Remove/replace old score_to_icon ---
score_to_icon = score_to_emoji

# --- New tick/cross indicator system ---
def score_to_ticks(score):
    """
    Converts a normalized score (0-1 or None) into a colored tick/cross/question mark HTML string.
    """
    green = '<span style="color:#1ca31c;font-weight:bold;">✔</span>'
    red = '<span style="color:#d11c1c;font-weight:bold;">✖</span>'
    purple = '<span style="color:#a020f0;font-weight:bold;">❔</span>'
    if score is None or score == -1 or (isinstance(score, float) and (np.isnan(score))):
        return purple
    if score >= 0.79:
        return green * 3
    elif score >= 0.59:
        return green * 2
    elif score >= 0.29:
        return green
    elif score >= 0.20:
        return red
    elif score >= 0.15:
        return red * 2
    else:
        return red * 3

# --- New plain tick/cross system for console output ---
def score_to_ticks_plain(score):
    """
    Converts a normalized score (0-1 or None) into plain Unicode ticks/crosses/question mark for console output.
    """
    if score is None or score == -1 or (isinstance(score, float) and (np.isnan(score))):
        return "❔"
    if score >= 0.75:
        return "✔✔✔"
    elif score >= 0.55:
        return "✔✔"
    elif score >= 0.3:
        return "✔"
    elif score >= 0.20:
        return "✖"
    elif score >= 0.10:
        return "✖✖"
    else:
        return "✖✖✖"

##############################
# Composite Scoring Function
##############################
def calculate_composite_score(
    row, weights, field_stats, todays_course, todays_distance, todays_going, todays_class, total_runners, debug=False
):
    # --- Individual Factors ---
    odds_fractional = str(row["Odds"]).strip().strip("'")
    odds_numeric = parse_fractional_odds(odds_fractional)
    odds_factor = 1 / odds_numeric if odds_numeric and odds_numeric > 0 else 0

    # --- Define category weights (normalize to sum to 1) ---
    suitability_weight = weights.get("going_suitability", 0) + weights.get("distance_suitability", 0) + weights.get("class", 0)
    form_weight = weights.get("recent_form", 0) + weights.get("last_ran", 0) + weights.get("comments", 0)
    market_weight = weights.get("odds", 0) + weights.get("stall", 0) + weights.get("headgear", 0)
    connections_weight = weights.get("jockey_trainer", 0) + weights.get("course", 0)
    raw_weight = weights.get("official_rating", 0) + weights.get("age", 0) + weights.get("weight_field", 0)

    total = suitability_weight + form_weight + market_weight + connections_weight + raw_weight
    if total == 0:
        suitability_weight = form_weight = market_weight = connections_weight = raw_weight = 0.2
    else:
        suitability_weight /= total
        form_weight /= total
        market_weight /= total
        connections_weight /= total
        raw_weight /= total

    try:
        official_rating = float(row["Official Rating"])
        official_rating_factor = official_rating / 100
    except:
        official_rating_factor = 0

    past_perf = parse_past_performance(row["Past Race History"], row["Race Date"])
    similar_perf = parse_similar_performance(
        row["Past Race History"], todays_course, todays_distance, todays_going, todays_class, row["Race Date"]
    )
    # Stall factor: Favor lower stall numbers in sprints, neutral otherwise
    try:
        stall_num = int(row["Stall"])
        if total_runners > 1:
            stall_factor_val = 1 - ((stall_num - 1) / (total_runners - 1))
        else:
            stall_factor_val = 0.5
    except:
        stall_factor_val = 0.5

    headgear_factor_val = parse_headgear_factor(row["Headgear"], row["Comments"])
    age_factor_val = age_factor(row["Age"], optimal=7, std=3)

    race_distance = field_stats.get("race_distance")
    if race_distance is None or race_distance <= 0:
        race_distance = 5000  # Default if missing

    recent_form_factor_val = recent_form_factor(row["Recent Form"])
    comments_factor_val = comments_sentiment_factor(row["Comments"])
    last_ran_factor_val = last_ran_factor(row["Last Ran (Days)"], race_distance)
    # --- Insert detection and override for unraced horses ---
    # Check if horse has ever raced (based on missing days and fallback recent)
    try:
        days_since_run = float(row["Last Ran (Days)"])
    except:
        days_since_run = None
    has_raced = not pd.isna(days_since_run) and recent_form_factor_val != 0.50
    if recent_form_factor_val == 0.50:
        comments_factor_val = min(comments_factor_val, 1.0) * 0.5  # Downscale generic comment boost
    if not has_raced:
        form_score = None  # Mark form as unknown
    else:
        form_score = (
            max(0, min(1, recent_form_factor_val))
            + max(0, min(1, comments_factor_val / 2))  # comments_factor_val can be >1, so scale
            + max(0, min(1, last_ran_factor_val))
        ) / 3

    market_score = (
        max(0, min(1, odds_factor / 10))  # odds_factor can be >1, so scale
        + max(0, min(1, stall_factor_val))
        + max(0, min(1, headgear_factor_val * 2))
    ) / 3

    # Calculate jt_factor (jockey/trainer combo) - placeholder logic, replace with your own as needed
    jt_factor = 0.5  # Default neutral value; replace with actual calculation if available

    # Calculate course_factor_val using the course_factor function
    course_factor_val = course_factor(
        row["Past Race History"], 
        row["Race Location"], 
        row["Race Date"]
    )

    connections_score = (
        max(0, min(1, jt_factor))
        + max(0, min(1, course_factor_val))
    ) / 2

    # Calculate weight_field_factor_val
    try:
        avg_weight = field_stats.get("avg_weight", 0)
        weight_field_factor_val = weight_factor(row["Weight"], avg_weight)
    except:
        weight_field_factor_val = 0.5

    raw_score = (
        max(0, min(1, official_rating_factor))
        + max(0, min(1, age_factor_val))
        + max(0, min(1, weight_field_factor_val))
    ) / 3

    # --- Calculate suitability_score before using it ---
    going_score = going_suitability(row["Past Race History"], todays_going, row["Race Date"])
    distance_score = distance_suitability(row["Past Race History"], todays_distance, row["Race Date"])
    class_score = class_factor(row["Past Race History"], todays_class, row["Race Date"])

    # Suitability score: average of available (not None) scores
    suitability_components = [s for s in [going_score, distance_score, class_score] if s is not None]
    suitability_score = sum(suitability_components) / len(suitability_components) if suitability_components else None

    # --- Final Composite Score (weighted sum of categories, then scaled up for legacy compatibility) ---
    # If suitability_score is None, reduce confidence (scale down composite)
    if suitability_score is not None:
        composite_suitability = suitability_score * suitability_weight
    else:
        composite_suitability = 0  # Or optionally: suitability_weight * 0.5

    # --- Fix: Use fallback if form_score is None ---
    safe_form_score = form_score if form_score is not None else 0.5

    composite_score = (
        composite_suitability
        + safe_form_score * form_weight
        + market_score * market_weight
        + connections_score * connections_weight
        + raw_score * raw_weight
    ) * 1000  # Scale for legacy output

    if suitability_score is None:
        composite_score *= 0.85  # Downgrade confidence if suitability unknown

    # Penalize longshots with poor past performance and low suitability
    if (
        odds_numeric is not None and odds_numeric > 10
        and past_perf < 0.4
        and suitability_score is not None and suitability_score < 0.4
    ):
        composite_score *= 0.85  # Reduce raw score

    # Noise floor for sparse data
    if row["Past Race History"] == "Unknown" or row["Past Race History"].count("||||") < 2:
        composite_score *= 0.9  # Unreliable profile

    # --- Scoring Tiers & Standout Flags ---
    tags = []
    if class_score is not None and class_score > 0.6:
        tags.append("🟢 High Class Match")
    if distance_score is not None and distance_score < 0.2:
        tags.append("🔴 Poor Distance Match")
    if jt_factor > 0.6 and course_factor_val > 0.7:
        tags.append("💡 Trainer Pattern Match")
    if last_ran_factor_val > 0.9 and suitability_score is not None and suitability_score > 0.8:
        tags.append("⚡ Fresh & Suited")

    # Add sub-scores for output table
    row["_suitability_score"] = suitability_score
    row["_form_score"] = form_score
    row["_market_score"] = market_score
    row["_connections_score"] = connections_score
    row["_raw_score"] = raw_score

    # --- Debug Output ---
    if debug:
        def fmt(x):
            return f"{x:.2f}" if x is not None else "❓"
        print(f"\n🔢 {row['Horse Name']}:")
        print(f"  [Suitability 40%]   Going: {fmt(going_score)}  Distance: {fmt(distance_score)}  Class: {fmt(class_score)}  → {fmt(suitability_score)}")
        print(f"  [Form 25%]         Recent: {fmt(recent_form_factor_val)}  Comments: {fmt(comments_factor_val)}  Last Ran: {fmt(last_ran_factor_val)}  → {fmt(form_score)}")
        print(f"  [Market 15%]       Odds: {fmt(odds_factor)}  Stall: {fmt(stall_factor_val)}  Headgear: {fmt(headgear_factor_val)}  → {fmt(market_score)}")
        print(f"  [Connections 10%]  Jockey/Trainer: {fmt(jt_factor)}  Course: {fmt(course_factor_val)}  → {fmt(connections_score)}")
        print(f"  [Raw 10%]          OR: {fmt(official_rating_factor)}  Age: {fmt(age_factor_val)}  Weight: {fmt(weight_field_factor_val)}  → {fmt(raw_score)}")
        print(f"  [Category Weights] Suitability: {suitability_weight*100:.0f}%, Form: {form_weight*100:.0f}%, Market: {market_weight*100:.0f}%, Connections: {connections_weight*100:.0f}%, Raw: {raw_weight*100:.0f}%")
        print(f"  [Composite Score]  {composite_score:.0f}   {' '.join(tags)}")

    return composite_score


##############################
# Custom Weight Input
##############################
def input_custom_weights():
    # Same defaults you had in __main__
    return {
        "going_suitability":       50,
        "distance_suitability":    50,
        "class":                   40,
        "past_performance":        45,
        "similar_conditions":      45,
        "recent_form":             25,
        "last_ran":                20,
        "odds":                    40,
        "official_rating":         40,
        "weight_field":            15,
        "stall":                   10,
        "headgear":                 5,
        "age":                      5,
        "comments":                 0,
        "course":                  15,
        "jockey_trainer":          15
    }


##############################
# Additional Helper Functions
##############################
def compute_field_stats(df):

    race_type_data = df.iloc[0]["Race Type Data"]
    print(f"🧾 Original Race Type Data: {race_type_data}")
    race_distance = get_todays_distance(race_type_data)
    print(f"🔍 Parsed distance: {race_distance} yards")

    weights = []
    ages = []
    for idx, row in df.iterrows():
        w = parse_weight_to_lbs(row["Weight"])
        if w and not np.isnan(w):
            weights.append(w)
        try:
            a = float(row["Age"])
            ages.append(a)
        except:
            pass
    avg_weight = sum(weights) / len(weights) if weights else 0
    avg_age = sum(ages) / len(ages) if ages else 0


    race_distance = get_todays_distance(df.iloc[0]["Race Type Data"])
    if race_distance is None or race_distance <= 0:
        race_distance = 5000

    return {"avg_weight": avg_weight, "avg_age": avg_age, "race_distance": race_distance}

def get_race_type(race_name):
    race_name_lower = race_name.lower()
    if "classified" in race_name_lower:
        return "Classified Stakes"
    elif "handicap" in race_name_lower:
        return "Handicap"
    elif "maiden" in race_name_lower:
        return "Maiden"
    else:
        return "Other"

##############################
# Main Modeling Functions
##############################
def model_race(csv_filename, weights):
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Fix: Check if DataFrame is empty before accessing any rows
    if df.empty:
        print("❌ ERROR: Race card CSV is empty. No runners were successfully scraped.")
        return

    # Get race name and extract type
    race_name = df.iloc[0]["Race Name"]
    course_name = df.iloc[0]["Race Location"]
    detailed_type = extract_race_type_from_name(race_name)
    print(f"🧬 Detected Race Type: {detailed_type}")

    # --- Belmont Stakes/Saratoga/Belmont detection ---
    if "Saratoga" in course_name or "Belmont Stakes" in race_name:
        is_belmont_stakes = True
    else:
        is_belmont_stakes = False

    # Compute field stats and race conditions early
    if df.empty:
        print("❌ ERROR: Race card CSV is empty. No runners were successfully scraped.")
        return

    field_stats = compute_field_stats(df)

    race_distance = field_stats.get("race_distance", 5000)
    if race_distance is None or race_distance <= 0:
        print("⚠️ race_distance not found, defaulting to 5000 yards.")
        race_distance = 5000
    #else:
        #print(f"📏 Race distance parsed: {race_distance} yards")

    todays_course = df.iloc[0]["Race Location"]
    todays_distance = get_todays_distance(df.iloc[0]["Race Type Data"])
    #print(f"🔍 Parsed distance from 'Race Type Data': {todays_distance} yards")
    todays_going = get_todays_going(df.iloc[0]["Race Type Data"])
    print(f"🧬 Detected Going: {todays_going}")
    going = todays_going or ""
    todays_class = parse_class_from_race_type(df.iloc[0]["Race Type Data"])
    total_runners = len(df)
    race_distance = field_stats.get("race_distance", 5000)

    # 🏇 York-specific adjustments
    if todays_course.lower() == "york":
        print("🎯 Applying York-specific model adjustments...")

        # 🔹 Sprint races (≤7f): strong stall bias and stamina matters
        if race_distance <= 1540:
            adjust_weight(weights, "stall", 1.3)
            adjust_weight(weights, "recent_form", 1.1)
            adjust_weight(weights, "past_performance", 1.1)

  # 🎯 Punchestown-specific adjustments
    if todays_course.lower() == "punchestown":
        print("🎯 Applying Punchestown-specific model adjustments...")
        rtype = detailed_type.lower()

        # 🔹 Novice/Chase (<3m): stamina & jockey/trainer combo (jump form proxy)
        if rtype.endswith("chase") and race_distance <= 5040:
            adjust_weight(weights, "distance_suitability", 1.2)
            adjust_weight(weights, "jockey_trainer",       1.2)

        # 🔹 Staying Chases (≥3m): extra stamina & class
        elif rtype.endswith("chase"):
            adjust_weight(weights, "distance_suitability", 1.3)
            adjust_weight(weights, "official_rating",      1.1)

        # 🔹 Hurdles (all trips): front-running stamina & going
        elif "hurdle" in rtype:
            adjust_weight(weights, "recent_form",         1.15)
            adjust_weight(weights, "going_suitability",   1.1)

        # Uphill finish demands class & robustness across NH
        adjust_weight(weights, "official_rating",      1.1)
        adjust_weight(weights, "weight_field",         1.05)
        # Course bias: proven on tight, undulating NH tracks
        adjust_weight(weights, "course",               1.1)


    # 🎯 Perth-specific adjustments
    if todays_course.lower() == "perth":
        print("🎯 Applying Perth-specific model adjustments...")
        # 🔹 Sprinters & small-field bias: inside draw & pace
        if race_distance <= 1320:
            adjust_weight(weights, "stall",           1.2)
            adjust_weight(weights, "pace_rating",     1.1)
        # 🔹 Middle trips (>6f–1¼m): uphill finish stamina
        elif race_distance <= 2200:
            adjust_weight(weights, "distance_suitability", 1.15)
            adjust_weight(weights, "official_rating",      1.1)
        else:
            # Stayers: extra stamina & going bias
            adjust_weight(weights, "distance_suitability", 1.2)
            adjust_weight(weights, "going_suitability",    1.1)
        # Course bias: tight, undulating left-handed
        adjust_weight(weights, "course",               1.05)


    # 🎯 ParisLongChamp-specific adjustments
    if todays_course.lower() in {"parislongchamp", "paris longchamp"}:
        print("🎯 Applying ParisLongChamp-specific model adjustments...")
        # 🔹 Sprints (≤7f): draw slightly matters on sweeping bends
        if race_distance <= 1400:
            adjust_weight(weights, "stall",           1.1)
            adjust_weight(weights, "recent_form",     1.05)
        # 🔹 Middle (7f–10f): gallop & class
        elif race_distance <= 2000:
            adjust_weight(weights, "distance_suitability", 1.15)
            adjust_weight(weights, "official_rating",      1.2)
        else:
            # Arc-distance & above: stamina & going
            adjust_weight(weights, "distance_suitability", 1.2)
            adjust_weight(weights, "going_suitability",    1.1)
        # 🔹 Big-field kick: use recent form as a proxy for pace in big fields
        if race_distance >= 2000:
            adjust_weight(weights, "recent_form",      1.1)
        # Course bias: proven on wide, galloping tracks
        adjust_weight(weights, "course",               1.1)


    if todays_course.lower() == "chelmsford city":
        print("🎯 Applying Chelmsford-specific model adjustments...")
        if race_distance <= 1760:  # 1 mile or less
            adjust_weight(weights, "stall", 1.3)
            adjust_weight(weights, "recent_form", 1.1)
            adjust_weight(weights, "past_performance", 1.1)
            adjust_weight(weights, "going_suitability", 1.1)

                # 🎯 Goodwood-specific adjustments
    if todays_course.lower() == "goodwood":
        print("🎯 Applying Goodwood-specific model adjustments...")
        # 🔹 Sprint (≤7f): bend bias → inside stalls, form counts
        if race_distance <= 1540:
            adjust_weight(weights, "stall", 1.3)
            adjust_weight(weights, "recent_form", 1.1)
            adjust_weight(weights, "past_performance", 1.1)
        # 🔹 Middle (7f–1¼m): uphill finish stamina & going
        elif race_distance <= 2200:
            adjust_weight(weights, "distance_suitability", 1.2)
            adjust_weight(weights, "going_suitability", 1.1)
        # 🔹 Stays (>1¼m): strong stamina & class signal
        else:
            adjust_weight(weights, "distance_suitability", 1.3)
            adjust_weight(weights, "going_suitability", 1.2)
        # Uphill finish demands class & robustness
        adjust_weight(weights, "official_rating", 1.2)
        adjust_weight(weights, "weight_field", 1.1)
        # Course bias: horses experienced at undulating/triangular tracks
        adjust_weight(weights, "course", 1.1)

        
    if todays_course.lower() == "nottingham":
        print("🎯 Applying Nottingham-specific model adjustments...")
        if race_distance <= 1320 and "soft" in going:
            adjust_weight(weights, "stall", 1.2)
        if "good to firm" in going or "firm" in going:
            adjust_weight(weights, "recent_form", 1.2)

    if todays_course.lower() == "fakenham":
        print("🎯 Applying Fakenham-specific model adjustments...")
        adjust_weight(weights, "jockey_trainer", 1.3)
        adjust_weight(weights, "recent_form", 1.2)
        adjust_weight(weights, "past_performance", 1.2)
        adjust_weight(weights, "similar_conditions", 1.2)

    if "chantilly" in todays_course.lower():
        print("🎯 Applying Chantilly-specific model adjustments...")
        adjust_weight(weights, "going_suitability", 1.2)
        adjust_weight(weights, "distance_suitability", 1.1)
        if race_distance <= 1760:
            adjust_weight(weights, "stall", 1.2)
        adjust_weight(weights, "class", 0.8)

    if todays_course.lower() == "kilbeggan":
        print("🎯 Applying Kilbeggan-specific model adjustments...")
        adjust_weight(weights, "past_performance", 1.2)
        adjust_weight(weights, "similar_conditions", 1.2)
        adjust_weight(weights, "jockey_trainer", 1.2)
        adjust_weight(weights, "class", 1.1)

    if todays_course.lower() == "listowel":
        print("🎯 Applying Listowel-specific model adjustments...")
        if "soft" in going or "heavy" in going:
            adjust_weight(weights, "going_suitability", 1.4)
            adjust_weight(weights, "distance_suitability", 1.2)
            adjust_weight(weights, "past_performance", 1.1)
        adjust_weight(weights, "class", 1.1)



        # 🔹 Long-distance: bend bias for wide draws
        if race_distance >= 2200 and total_runners >= 12:
            adjust_weight(weights, "stall", 1.2)

        # 🔹 General York traits: class and exposed form more predictive
        adjust_weight(weights, "class", 1.3)
        adjust_weight(weights, "official_rating", 1.2)
        adjust_weight(weights, "past_performance", 1.2)



    # Adjust weights based on race type
    race_type = get_race_type(df.iloc[0]["Race Name"])

    detailed_type = extract_race_type_from_name(df.iloc[0]["Race Name"])
    print(f"🧬 Detected Race Type: {detailed_type}")

    # Adjust weightings based on detailed race type
    if detailed_type == "Novice Hurdle":
        adjust_weight(weights, "recent_form", 1.2)
        adjust_weight(weights, "similar_conditions", 1.2)
        adjust_weight(weights, "jockey_trainer", 1.2)
        adjust_weight(weights, "comments", 1.1)
    elif detailed_type == "Novice Chase":
        adjust_weight(weights, "recent_form", 1.2)
        adjust_weight(weights, "similar_conditions", 1.1)
        adjust_weight(weights, "jockey_trainer", 1.2)
    elif detailed_type == "Handicap":
        adjust_weight(weights, "official_rating", 1.3)
        adjust_weight(weights, "weight_field", 1.2)
    elif detailed_type == "Group 1":
        adjust_weight(weights, "past_performance", 1.3)
        adjust_weight(weights, "similar_conditions", 1.2)
        adjust_weight(weights, "jockey_trainer", 1.2)
    elif detailed_type == "Bumper":
        adjust_weight(weights, "age", 1.2)
        adjust_weight(weights, "recent_form", 1.1)
    elif detailed_type == "Maiden":
        adjust_weight(weights, "comments", 1.2)
        adjust_weight(weights, "headgear", 1.2)
    elif detailed_type == "Juvenile Hurdle":
        adjust_weight(weights, "age", 1.3)
        adjust_weight(weights, "recent_form", 1.1)
    elif detailed_type == "Veterans Handicap Chase":
        adjust_weight(weights, "official_rating", 1.2)
        adjust_weight(weights, "age", 1.3)
    elif detailed_type == "Maiden":
        adjust_weight(weights, "headgear", 1.2)
        # Defensive: only set if present
        if "trainer_jockey_combo" in weights:
            adjust_weight(weights, "trainer_jockey_combo", 1.2)
    elif detailed_type == "Classified Stakes":
        adjust_weight(weights, "official_rating", 1.1)
        adjust_weight(weights, "recent_form", 1.1)
    elif detailed_type == "Claiming":
        adjust_weight(weights, "weight_field", 1.1)
        # Defensive: only set if present
        if "trainer_jockey_combo" in weights:
            adjust_weight(weights, "trainer_jockey_combo", 1.1)

    # Distance band logic (trip type)
    if race_distance <= 1320:  # Sprint: ≤6f
        adjust_weight(weights, "recent_form", 1.2)
        adjust_weight(weights, "stall", 1.3)
        adjust_weight(weights, "last_ran", 1.1)
    elif 1321 <= race_distance <= 2200:  # Intermediate: 7f–9f
        adjust_weight(weights, "distance_suitability", 1.1)
    elif 2201 <= race_distance <= 2900:  # Middle-distance: 1m2f–1m4f
        adjust_weight(weights, "distance_suitability", 1.2)
        adjust_weight(weights, "going_suitability", 1.1)
    elif race_distance > 2900:  # Staying race
        adjust_weight(weights, "distance_suitability", 1.3)
        adjust_weight(weights, "going_suitability", 1.3)
        adjust_weight(weights, "past_performance", 1.2)
        adjust_weight(weights, "last_ran", 1.2)

    # Going condition logic
    going = todays_going or ""
    if "soft" in going or "heavy" in going:
        adjust_weight(weights, "going_suitability", 1.4)
        adjust_weight(weights, "class", 1.1)
        adjust_weight(weights, "past_performance", 1.1)
    elif "firm" in going or "good to firm" in going:
        adjust_weight(weights, "stall", 1.2)
        adjust_weight(weights, "recent_form", 1.1)

    # Tactical race shape bias
    if total_runners >= 12 and race_distance <= 1320 and "firm" in going:
        adjust_weight(weights, "stall", 1.3)
        adjust_weight(weights, "recent_form", 1.2)

    if total_runners <= 3:
        adjust_weight(weights, "jockey_trainer", 1.3)
        adjust_weight(weights, "recent_form", 1.3)

    # Set weight_field to 0 if all weights are identical
    weight_values = df["Weight"].unique()
    if len(weight_values) == 1:
        weights["weight_field"] = 0


    # 🧪 Class performance impact
    for index, row in df.iterrows():
        direction, experience_score, inexperience_penalty = assess_class_change(row, todays_class)

        if direction == "up":
            df.at[index, "class_factor"] = -1 * (inexperience_penalty * 2)  # Penalize up in class
        elif direction == "down":
            df.at[index, "class_factor"] = experience_score * 1.5  # Reward class drop
        else:
            df.at[index, "class_factor"] = experience_score  # Mild reward for experience

    # Example integration: apply class_factor to model score
    df["class_factor"] = df["class_factor"].fillna(0)

    # Adjust weightings based on detected race type
    if detailed_type == "Novice Hurdle":
        weights["recent_form"] *= 1.2
        weights["similar_conditions"] *= 1.2
        weights["jockey_trainer"] *= 1.2
        weights["comments"] *= 1.1
    elif detailed_type == "Novice Chase":
        weights["recent_form"] *= 1.2
        weights["similar_conditions"] *= 1.1
        weights["jockey_trainer"] *= 1.2
    elif detailed_type == "Handicap":
        weights["official_rating"] *= 1.3
        weights["weight_field"] *= 1.2
    elif detailed_type == "Group 1":
        weights["past_performance"] *= 1.3
        weights["similar_conditions"] *= 1.2
        weights["jockey_trainer"] *= 1.2
    elif detailed_type == "Bumper":
        weights["age"] *= 1.2
        weights["recent_form"] *= 1.1
    elif detailed_type == "Maiden":
        weights["comments"] *= 1.2
        weights["headgear"] *= 1.2
    elif detailed_type == "Juvenile Hurdle":
        weights["age"] *= 1.3
        weights["recent_form"] *= 1.1
    elif detailed_type == "Veterans Handicap Chase":
        weights["official_rating"] *= 1.2
        weights["age"] *= 1.3
    elif detailed_type == "Maiden":
        weights["headgear"] *= 1.2
        # Defensive: only set if present
        if "trainer_jockey_combo" in weights:
            weights["trainer_jockey_combo"] *= 1.2
    elif detailed_type == "Classified Stakes":
        weights["official_rating"] *= 1.1
        weights["recent_form"] *= 1.1
    elif detailed_type == "Claiming":
        weights["weight_field"] *= 1.1
        # Defensive: only set if present
        if "trainer_jockey_combo" in weights:
            weights["trainer_jockey_combo"] *= 1.1

    # Calculate composite scores with debug output
    print("\n🧪 Composite Score Debug Breakdown:")

    # Collect all sub-scores for each row
    def scoring_wrapper(row):
        # --- Tactical bonus for Epsom Downs sprints ---
        tactical_bonus = 0.0
        if todays_course.lower() == "epsom downs" and race_distance is not None and race_distance <= 1320:
            tactical_bonus += row.get("Draw Bonus", 0.0)
            tactical_bonus += row.get("Pace Bonus", 0.0)
        score = calculate_composite_score(
            row, weights, field_stats, todays_course, todays_distance, todays_going, todays_class, total_runners, debug=True
        )
        # --- Doncaster tactical bonuses ---
        if todays_course.lower() == "doncaster":
            # Hold-up bias for big-field straight mile+
            try:
                distance_furlongs = race_distance / 220 if race_distance else 0
                field_size = total_runners
                pace_style = str(row.get("Comments", "")).lower()
                # Simple proxy for hold-up/mid-division
                if distance_furlongs >= 8 and field_size >= 12 and any(x in pace_style for x in ["held up", "mid-division", "waited", "waited with", "towards rear", "rear"]):
                    score += 0.1 * score
                # Low draw bonus for sprints
                try:
                    draw = int(row.get("Stall", 99))
                    if distance_furlongs <= 6 and field_size >= 12 and draw <= 8:
                        score += 0.05 * score
                except Exception:
                    pass
                # Trainer bonus
                if "trainer_bonus" in row and row["trainer_bonus"]:
                    score += float(row["trainer_bonus"]) * score
            except Exception:
                pass
        # Add tactical bonus (as a percentage of the score)
        score = score * (1 + tactical_bonus)
        # --- Belmont Stakes logic ---
        if is_belmont_stakes:
            # Reward stamina indicators more
            # Upweight distance suitability if available
            distance_score = distance_suitability(row["Past Race History"], todays_distance, row["Race Date"])
            if distance_score is not None:
                distance_score *= 1.15
            # Penalize horses with no previous dirt form or stamina signs
            going_pref = "dirt" if "dirt" in str(row.get("Comments", "")).lower() or "dirt" in str(row.get("Past Race History", "")).lower() else "other"
            max_distance_run = 0
            # Try to extract max distance run in furlongs from past history
            try:
                import re
                dists = re.findall(r"Distance:\s*([\d ]*m)?\s*([\d ]*f)?\s*([\d ]*y)?", row.get("Past Race History", ""))
                for m, f, y in dists:
                    miles = int(m.strip().replace("m", "")) if m.strip() else 0
                    furlongs = int(f.strip().replace("f", "")) if f.strip() else 0
                    # ignore yards for this logic
                    total_f = miles * 8 + furlongs
                    if total_f > max_distance_run:
                        max_distance_run = total_f
            except Exception:
                pass
            suitability_penalty = 1.0
            if going_pref != "dirt":
                suitability_penalty *= 0.9
            stamina_flag = False
            if max_distance_run < 10:
                stamina_flag = True
                if distance_score is not None:
                    distance_score *= 0.8
            # Trainer bias
            trainer_name = str(row.get("Trainer", ""))
            trainer_score = 1.0
            if trainer_name in ["Todd Pletcher", "Bob Baffert", "Brad Cox"]:
                trainer_score *= 1.1
            # Special note
            row["special_note"] = "🏆 Belmont Stakes logic applied"
            # Optionally, you could inject these modifiers into the row for further use
            row["_distance_score_belmont"] = distance_score
            row["_trainer_score_belmont"] = trainer_score
            row["_suitability_penalty_belmont"] = suitability_penalty
            row["_stamina_flag_belmont"] = stamina_flag

        # Return all sub-scores as a tuple
        return (
            score,
            row.get("_suitability_score", None),
            row.get("_form_score", None),
            row.get("_market_score", None),
            row.get("_connections_score", None),
            row.get("_raw_score", None)
        )

    # Apply and unpack results
    results = df.apply(scoring_wrapper, axis=1, result_type='expand')
    df["Composite Score"] = results[0]
    df["_suitability_score"] = results[1]
    df["_form_score"] = results[2]
    df["_market_score"] = results[3]
    df["_connections_score"] = results[4]
    df["_raw_score"] = results[5]
    df["Composite Score"] = df["Composite Score"].fillna(0)

    # Ensure all sub-score columns exist before using them
    for col in ["_suitability_score", "_form_score", "_market_score", "_connections_score", "_raw_score"]:
        if col not in df.columns:
            print(f"❌ Error: '{col}' column missing — was scoring applied correctly?")
            return

    df_sorted = df.sort_values(by="Composite Score", ascending=False).reset_index(drop=True)

    # Define the odds ladder
    odds_ladder = [
        '1/100', '1/80', '1/66', '1/50', '1/33', '1/25', '1/20', '1/16', '1/14', '1/12', '1/10', '1/8', '1/7', '1/6', '1/5', '1/4', '2/7', '1/3', '2/5', '1/2', '4/7', '8/13', '4/6', '8/11', '4/5', '5/6', '10/11', '1/1', '11/10', '6/5', '5/4', '11/8', '6/4',
        '13/8', '7/4', '15/8', '2/1', '9/4', '5/2', '11/4', '3/1', '10/3', '7/2',
        '4/1', '9/2', '5/1', '11/2', '6/1', '13/2', '7/1', '15/2', '8/1', '9/1',
        '10/1', '11/1', '12/1', '14/1', '16/1', '20/1', '25/1', '28/1', '33/1', '40/1', '50/1', '66/1', '80/1', '100/1', '125/1', '150/1', '200/1', '250/1', '300/1', '500/1', '1000/1'
    ]

    # Convert ladder to decimal odds for comparison
    ladder_decimal = [parse_fractional_odds_to_decimal(odds) for odds in odds_ladder]

    # Calculate bookmaker overround
    real_odds = df_sorted["Odds"].apply(lambda x: x.strip("'"))
    bookie_probs = [1 / parse_fractional_odds_to_decimal(odd) for odd in real_odds]
    bookie_overround = sum(bookie_probs)

    # Ladder system for modeled odds
    mv_ranks = df_sorted["Composite Score"].rank(ascending=False, method='min')
    total_horses = len(df_sorted)
    max_steps = 8

    modeled_odds_decimal = []
    modeled_odds_fractional = []
    real_odds_indices = []
    modeled_odds_indices = []
    for idx, row in df_sorted.iterrows():
        real_odd = row["Odds"].strip("'")
        real_odd_decimal = parse_fractional_odds_to_decimal(real_odd)
        closest_idx = min(range(len(ladder_decimal)), key=lambda i: abs(ladder_decimal[i] - real_odd_decimal))
        real_odds_indices.append(closest_idx)
        
        rank = mv_ranks[idx]
        relative_position = (total_horses - rank + 2) / total_horses
        steps = int(round((relative_position - 0.5) * 2 * max_steps))
        new_idx = closest_idx - steps
        new_idx = max(0, min(len(odds_ladder) - 2, new_idx))
        
        modeled_odd_decimal = ladder_decimal[new_idx]
        modeled_odd_fractional = odds_ladder[new_idx]
        modeled_odds_decimal.append(modeled_odd_decimal)
        modeled_odds_fractional.append(modeled_odd_fractional)
        modeled_odds_indices.append(new_idx)

    df_sorted["Modelled Odds"] = modeled_odds_decimal
    df_sorted["Modelled Odds Fraction"] = modeled_odds_fractional

    # Calculate modeled odds overround (before calibration)
    modeled_probs = [1 / odd for odd in modeled_odds_decimal]
    modeled_overround = sum(modeled_probs)

    # Calibrate modeled odds to match bookmaker overround
    calibration_factor = bookie_overround / modeled_overround
    calibrated_probs = [prob * calibration_factor for prob in modeled_probs]
    calibrated_modeled_odds = [1 / prob for prob in calibrated_probs]
    df_sorted["Calibrated Modelled Odds"] = calibrated_modeled_odds

    # Add Calibrated Fraction Odds and store indices
    calibrated_fraction_odds = []
    calibrated_fraction_indices = []
    for calibrated_odd in calibrated_modeled_odds:
        closest_idx = min(range(len(ladder_decimal)), key=lambda i: abs(ladder_decimal[i] - calibrated_odd))
        calibrated_fraction_odds.append(odds_ladder[closest_idx])
        calibrated_fraction_indices.append(closest_idx)
    df_sorted["Calibrated Fraction Odds"] = calibrated_fraction_odds

    # Helper to convert sub-scores to a label for display
    def label_score(score):
        # Less harsh thresholds
        if score >= 0.8:
            return "JP"
        elif score >= 0.5:
            return "🟢"
        elif score >= 0.38:
            return "🟠"
        else:
            return "🔴"

    # Helper function to interpret scores for display
    def interpret_score(score, category):
        """
        Converts a normalized score (0-1) into a label or emoji for display.
        """
        if category == "going":
            if score >= 0.75:
                return "JP"
            elif score >= 0.5:
                return "🟢"
            elif score >= 0.38:
                return "🟠"
            else:
                return "🔴"
        elif category == "distance":
            if score >= 0.8:
                return "JP"
            elif score >= 0.5:
                return "🟢"
            elif score >= 0.38:
                return "🟠"
            else:
                return "🔴"
        elif category == "class":
            if score >= 0.8:
                return "JP"
            elif score >= 0.5:
                return "🟢"
            elif score >= 0.38:
                return "🟠"
            else:
                return "🔴"
        elif category == "form":
            if score >= 0.8:
                return "JP"
            elif score >=  0.5:
                return "🟢"
            elif score >= 0.38:
                return "🟠"
            else:
                return "🔴"
        else:
            if score >= 0.8:
                return "JP"
            elif score >= 0.5:
                return "🟢"
            elif score >= 0.38:
                return "🟠"
            else:
                return "🔴"

    # Helper function to clamp values between 0 and 1
    def clamp01(x):
        if x is None:
            return None
        return max(0, min(1, x))

    # Add separate columns for Going, Distance, and Class using tick/cross system
    df_sorted["Going"] = df_sorted.apply(
        lambda row: score_to_ticks(
            clamp01(going_suitability(row["Past Race History"], todays_going, row["Race Date"]))
        ), axis=1
    )
    df_sorted["Distance"] = df_sorted.apply(
        lambda row: score_to_ticks(
            clamp01(distance_suitability(row["Past Race History"], todays_distance, row["Race Date"]))
        ), axis=1
    )
    df_sorted["Class"] = df_sorted.apply(
        lambda row: score_to_ticks(
            clamp01(class_factor(row["Past Race History"], todays_class, row["Race Date"]))
        ), axis=1
    )
    # --- Use "❔" for unraced horses in Form column, matching other indicators ---
    df_sorted["Form"] = df_sorted["_form_score"].apply(lambda score: "❔" if score is None or pd.isna(score) else score_to_ticks(score))


    
    # Prepare enhanced output DataFrame (hide Market, Connections, Raw Stats, Suitability)
    output_df = df_sorted[
        [
            "Horse Name",
            "Odds",
            "Calibrated Fraction Odds",
            "Composite Score",
            "Going",
            "Distance",
            "Class",
            "Form"
        ]
    ].copy()

    # Round the composite score to an integer
    output_df["Composite Score"] = output_df["Composite Score"].round(0).astype(int)

    # Add fair odds (Model 100%) next to Real Odds
    model_win_prob = 1 / df_sorted["Calibrated Modelled Odds"]
    model_prob_sum = model_win_prob.sum()
    model_fair_prob = model_win_prob / model_prob_sum
    model_100_odds = (1 / model_fair_prob).round(2)
    output_df["Model 100%"] = model_100_odds.values

    # Compute indices for Model 100% odds on the fractional ladder
    model100_fraction_indices = []
    for odd in model_100_odds.values:
        closest_idx = min(range(len(ladder_decimal)), key=lambda j: abs(ladder_decimal[j] - odd))
        model100_fraction_indices.append(closest_idx)

    # Flag value bets based on Model 100% ladder positions
    def flag_value(row, real_idx, model_idx):
        # Positive if Model 100% snapped odds are shorter (lower index) by 5+ steps
        steps_shortened = real_idx - model_idx
        # Positive if Model 100% snapped odds are longer (higher index) by 5+ steps
        steps_worsened = model_idx - real_idx
        if steps_shortened >= 2:
            return "💰"
        elif steps_worsened >= 6:
            return "✖️"
        return ""

    output_df["💰 Value"] = [
        flag_value(row, real_odds_indices[i], model100_fraction_indices[i])
        for i, row in output_df.iterrows()
    ]

    # Rename for display
    # Rename for display
    output_df = output_df.rename(columns={
        "Horse Name": "🏇 Horse Name",
        "Odds": "🎯 Real Odds",
        "Calibrated Fraction Odds": "📊 Model Odds",
        "Composite Score": "📈 Score"
    })

    # Convert Real Odds to decimal
    output_df["🎯 Real Odds"] = output_df["🎯 Real Odds"] \
        .apply(lambda x: parse_fractional_odds_to_decimal(str(x)))
    output_df["🎯 Real Odds"] = output_df["🎯 Real Odds"].map(lambda d: f"{d:.2f}")

    # Reorder columns so Model 100% appears after Real Odds
    desired = [
        "🏇 Horse Name",
        "🎯 Real Odds",
        "Model 100%",
        "📊 Model Odds",
        "📈 Score",
        "Going",
        "Distance",
        "Class",
        "Form",
        "💰 Value"
    ]
    output_df = output_df[desired]



    # --- NEW: Save value bets to JSON ---
    def append_value_bet(bet):
        import json
        import os
        filename = "value_bets.json"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        else:
            data = []
        data.append(bet)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    for idx, row in output_df.iterrows():
        if row["💰 Value"] == "💰":
            # Gather info for the value bet
            orig_row      = df_sorted.iloc[idx]
            race_date     = orig_row["Race Date"]
            race_time     = orig_row["Race Time"]
            race_datetime = f"{race_date} {race_time}"
            course        = orig_row["Race Location"]
            horse         = row["🏇 Horse Name"]
            bookie_odds   = row["🎯 Real Odds"]
            # Pull the true decimal Model 100% odds instead of the snapped fractional odds
            model_odds    = row["Model 100%"]
            # --- Clean odds formatting ---
            bookie_odds = str(bookie_odds).strip().replace("'", "")
            # Ensure two‐decimal display
            model_odds  = f"{float(model_odds):.2f}"
            # Save value bet
            append_value_bet({
                "race_datetime": race_datetime,
                "course":        course,
                "horse":         horse,
                "bookie_odds":   bookie_odds,
                "model_odds":    model_odds
            })


    # Print enhanced table (update header accordingly)
    print("-" * 110)
    print(f"{'🏇 Horse Name':<22} {'🎯 Real Odds':<10} {'📊 Model Odds':<12} {'Model 100%':<10} {'📈 Score':<8} {'Going':<10} {'Distance':<10} {'Class':<10} {'Form':<10} {'💰 Value':<8}")
    print("-" * 110)
    for idx, row in output_df.iterrows():
        # Use plain ticks/crosses for console output
        orig_row = df_sorted.iloc[idx]
        going_plain = score_to_ticks_plain(clamp01(going_suitability(orig_row["Past Race History"], todays_going, orig_row["Race Date"])))
        distance_plain = score_to_ticks_plain(clamp01(distance_suitability(orig_row["Past Race History"], todays_distance, orig_row["Race Date"])))
        class_plain = score_to_ticks_plain(clamp01(class_factor(orig_row["Past Race History"], todays_class, orig_row["Race Date"])))
        form_plain = "❔" if orig_row["_form_score"] is None or pd.isna(orig_row["_form_score"]) else score_to_ticks_plain(orig_row["_form_score"])
        print(f"{row['🏇 Horse Name']:<22} {row['🎯 Real Odds']:^10} {row['📊 Model Odds']:^12} {str(row['Model 100%']):^10} {str(row['📈 Score']):^8} {going_plain:^10} {distance_plain:^10} {class_plain:^10} {form_plain:^10} {row['💰 Value']:^8}")

    print("\n" * 2)
    print(f"Bookmaker Overround: {bookie_overround * 100:.2f}%")
    print(f"Calibrated Modelled Overround: {sum(1 / odd for odd in calibrated_modeled_odds) * 100:.2f}%")
    print("=============================================\n")

    # --- Race Visualizer Integration ---
    # Prepare DataFrame for visualizer
    visual_df = df_sorted[["Horse Name", "Odds", "Calibrated Fraction Odds", "Composite Score"]].copy()
    visual_df = visual_df.rename(columns={
        "Horse Name": "🏇 Horse Name",
        "Odds": "🎯 Real Odds",
        "Calibrated Fraction Odds": "📊 Model Odds",
        "Composite Score": "📈 Score"
    })
    # Add value column for visualizer
    visual_df["💰 Value"] = output_df["💰 Value"]

    # Create race-specific filename and title
    race_time = df_sorted.iloc[0]["Race Time"].replace(":", "")
    race_name = df_sorted.iloc[0]["Race Name"].replace(" ", "_").replace("/", "-")
    html_output_path = f"race_visual_{race_name}_{race_time}.html"
    race_time_course = f"{df_sorted.iloc[0]['Race Time']} {df_sorted.iloc[0]['Race Location']}"

    # --- Stylized HTML Visualizer ---
    # def generate_race_visualizer(df, race_time_course="", output_file="race_prediction_visual.html"):
    #     """
    #     Generates a stylized HTML file showing horses and their predicted finishing positions.
    #     """
    #     html = f"""
    #     <html>
    #     <head>
    #         <style>
    #             body {{
    #                 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    #                 background: linear-gradient(to bottom, #f0f9ff, #cbebff);
    #                 padding: 40px;
    #                 color: #333;
    #             }}
    #             h1 {{
    #                 text-align: center;
    #                 margin-bottom: 5px;
    #                 color: #222;
    #             }}
    #             h2 {{
    #                 text-align: center;
    #                 margin-bottom: 30px;
    #                 font-weight: normal;
    #                 color: #444;
    #             }}
    #             table {{
    #                 margin: 0 auto;
    #                 border-collapse: collapse;
    #                 width: 95%;
    #                 max-width: 900px;
    #                 box-shadow: 0 0 10px rgba(0,0,0,0.1);
    #                 background: white;
    #             }}
    #             th, td {{
    #                 padding: 12px 15px;
    #                 text-align: center;
    #                 border: 1px solid #ccc;
    #                 font-size: 1em;
    #             }}
    #             th {{
    #                 background-color: #007acc;
    #                 color: white;
    #             }}
    #             tr:nth-child(even) {{
    #                 background-color: #f9f9f9;
    #             }}
    #             tr:hover {{
    #                 background-color: #eef;
    #             }}
    #         </style>
    #     </head>
    #     <body>
    #         <h1>🏇 Race Prediction Visualizer</h1>
    #         <h2>{race_time_course}</h2>
    #         <table>
    #             <tr>
    #                 <th>📊 Rank</th>
    #                 <th>🐎 Horse Name</th>
    #                 <th>🎯 Real Odds</th>
    #                 <th>📊 Model Odds</th>
    #                 <th>📈 Score</th>
    #                 <th>Going</th>
    #                 <th>Distance</th>
    #                 <th>Class</th>
    #                 <th>Form</th>
    #                 <th>💰 Value</th>
    #             </tr>
    #     """

    #     for idx, row in df.iterrows():
    #         rank = idx + 1
    #         horse = row.get("🏇 Horse Name", "Unknown")
    #         real_odds = row.get("🎯 Real Odds", "-")
    #         model_odds = row.get("📊 Model Odds", "-")
    #         score = int(round(row.get("📈 Score", 0)))
    #         value = row.get("💰 Value", "")

    #         # --- Use score_to_emoji for each component ---
    #         going_score = row.get("_suitability_score", None)
    #         distance_score = row.get("_suitability_score", None)
    #         class_score = row.get("_suitability_score", None)
    #         form_score = row.get("_form_score", None)
    #         going_icon = score_to_emoji(going_score)
    #         distance_icon = score_to_emoji(distance_score)
    #         class_icon = score_to_emoji(class_score)
    #         # --- Use "❓" for unknown form, matching other indicators ---
    #         import pandas as pd
    #         form_indicator = "❓" if form_score is None or pd.isna(form_score) else (
    #             "🟢" if form_score > 0.65 else "🟡" if form_score > 0.45 else "🔴"
    #         )

    #         html += f"""
    #             <tr>
    #                 <td>{rank}</td>
    #                 <td>{horse}</td>
    #                 <td>{real_odds}</td>
    #                 <td>{model_odds}</td>
    #                 <td>{score}</td>
    #                 <td>{going_icon}</td>
    #                 <td>{distance_icon}</td>
    #                 <td>{class_icon}</td>
    #                 <td>{form_indicator}</td>
    #                 <td>{value}</td>
    #             </tr>
    #         """

    #     html += """
    #         </table>
    #     </body>
    #     </html>
    #     """

    #     with open(output_file, "w", encoding="utf-8") as f:
    #         f.write(html)
    #     print(f"✅ Race visualization saved as: {output_file}")

    # # Call the new visualizer with race time & course
    # generate_race_visualizer(
    #     df=visual_df,
    #     race_time_course=race_time_course,
    #     output_file=html_output_path
    # )
    # print(f"\n📸 Race Visualizer Created: {html_output_path}\nOpen it in a browser to view or share.")

    # Save styled HTML for the race (sanitize filename)
    race_time = df_sorted.iloc[0]["Race Time"]
    race_course = df_sorted.iloc[0]["Race Location"]
    race_title = df_sorted.iloc[0]["Race Name"]
    safe_race_time = race_time.replace(":", "-")
    safe_race_course = race_course.replace(" ", "_")
    html_filename = f"{safe_race_time}_{safe_race_course}.html"
    # Drop Model Odds for HTML output
    html_df = output_df.drop(columns=["📊 Model Odds"])
    save_styled_html(html_df, f"{race_time} {race_course} — {race_title}", html_filename)

    # Track all generated HTML files for index
    if not hasattr(model_race, "_html_files"):
        model_race._html_files = []
    model_race._html_files.append(html_filename)
    update_index_html(model_race._html_files)

    # --- Upload to GitHub after each race HTML and index generation ---
#     upload_to_github(html_filename, html_filename)
#     upload_to_github("index.html", "index.html")
#     upload_to_github("value_bets.json", "value_bets.json")

def save_styled_html(df, race_title: str, filename: str):
    html_content = f"""
<html>
<head>
    <title>{race_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5; }}
        h1 {{ text-align: center; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: white; }}
        th, td {{ padding: 12px; text-align: center; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #333; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}

        /* ▶︎ Our new highlight classes */
        .value-horse {{ background-color: gold !important; }}
        .undervalued {{ background-color: lightblue !important; }}
    </style>
</head>
<body>
    <h1>{race_title}</h1>
    {df.to_html(index=False, escape=False)}
    <script>
    // After the table is rendered, walk each row …
    document.querySelectorAll('table tbody tr').forEach(row => {{
      const cells = row.querySelectorAll('td');
      const horseCell = cells[0];
      const realOdds = parseFloat(cells[1].textContent) || 0;
      const modelOdds = parseFloat(cells[2].textContent) || 0;
      const valueText = cells[cells.length - 1].textContent;  // last column

      // 1) gold if “Value” column contains our indicator  
      if (valueText.includes('💰') || valueText.includes('✔')) {{
        horseCell.classList.add('value-horse');
      }}
      // 2) blue if long odds but model thinks cheaper
      else if (realOdds >= 10 && modelOdds < realOdds) {{
        horseCell.classList.add('undervalued');
      }}
    }});
    </script>
</body>
</html>
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)


def update_index_html(page_links):
    from pathlib import Path

    # Always use the absolute path for index.html
    index_path = Path(r"c:\Users\scott\Documents\PythonScripts\index.html")
    # Load existing content if index exists
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        # Start fresh if no index exists
        content = """
        <html>
        <head>
            <title>Race Visualizer</title>
            <style>
                body { font-family: Arial, sans-serif; background-color: #f5f5f5; padding: 30px; }
                h1 { text-align: center; }
                ul { list-style-type: none; padding: 0; }
                li { margin: 12px 0; font-size: 18px; }
                a { text-decoration: none; color: #007bff; }
            </style>
        </head>
        <body>
            <h1>Race Visualizer</h1>
            <ul>
            </ul>
        </body>
        </html>
        """

    # Find the current <ul>...</ul> block
    import re
    ul_match = re.search(r"(<ul>.*?</ul>)", content, re.DOTALL)
    if ul_match:
        ul_block = ul_match.group(1)
    else:
        ul_block = "<ul>\n</ul>"

    # Extract existing links
    existing_links = set(re.findall(r"href=['\"]([^'\"]+)['\"]", ul_block))

    # Add new links if not already present
    new_entries = []
    for link in page_links:
        if link not in existing_links:
            entry = f"<li><a href='{link}'>{link.replace('.html', '')}</a></li>"
            new_entries.append(entry)
            existing_links.add(link)

    # Insert new entries before </ul>
    if new_entries:
        ul_block = ul_block.replace("</ul>", "\n" + "\n".join(new_entries) + "\n</ul>")

    # Replace the old <ul>...</ul> with the updated one
    content = re.sub(r"<ul>.*?</ul>", ul_block, content, flags=re.DOTALL)

    # Save the updated index.html
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(content)

def generate_index_html(output_dir="."):
    import os
    # Always save index.html to the fixed path
    index_html_path = r"c:\Users\scott\Documents\PythonScripts\index.html"
    html_files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith(".html") and f != "index.html"]
    )

    # Organize by course
    course_map = {}
    for filename in html_files:
        try:
            time_part, course_part = filename.replace(".html", "").split("_", 1)
            course = course_part
            course_map.setdefault(course, []).append((time_part, filename))
        except Exception:
            print(f"Skipping malformed filename: {filename}")

    # Build HTML content
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Race Visualizer</title>
  <style>
    body { margin: 0; font-family: 'Segoe UI', sans-serif; background: #1e1e2f; color: #f5f5f5; display: flex; flex-direction: column; align-items: center; }
    header { background: linear-gradient(135deg, #222 40%, #444); width: 100%; padding: 2rem 1rem; text-align: center; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6); }
    header h1 { margin: 0; font-size: 2.5rem; color: #ffd700; letter-spacing: 1px; }
    .container { width: 100%; max-width: 1200px; padding: 2rem; }
    .container h2 { width: 100%; text-align: left; margin: 2rem 0 1rem; color: #ffd700; border-bottom: 1px solid #444; padding-bottom: 0.3rem; }
    .course-group { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }
    .card { background: #2c2c3e; border-radius: 12px; padding: 1.5rem; text-align: center; width: 220px; transition: transform 0.2s ease, box-shadow 0.3s ease; box-shadow: 0 0 10px rgba(255, 255, 255, 0.05); }
    .card:hover { transform: translateY(-6px); box-shadow: 0 0 15px rgba(255, 215, 0, 0.4); }
    .card a { text-decoration: none; color: #ffd700; font-weight: bold; font-size: 1.1rem; }
    footer { margin-top: 3rem; font-size: 0.9rem; color: #888; text-align: center; padding-bottom: 2rem; }
    @media (max-width: 600px) { .card { width: 90%; } header h1 { font-size: 1.8rem; } }
  </style>
</head>
<body>
  <header>
    <h1>🏇 Race Visualizer Index</h1>
  </header>
  <div class="container">
'''

    for course, entries in sorted(course_map.items()):
        html_content += f'    <h2>🏁 {course}</h2>\n    <div class="course-group">\n'
        for time, fname in entries:
            time_display = time.replace("-", ":")
            html_content += f'      <div class="card"><a href="{fname}">{time_display} {course}</a></div>\n'
        html_content += '    </div>\n'

    html_content += '''  </div>
  <footer>
    &copy; 2025 ScottyMelloty Racing Visualizer • All rights reserved
  </footer>
</body>
</html>'''

    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ index.html regenerated.")

def upload_to_github(file_path, github_path):
    import os
    from dotenv import load_dotenv
    load_dotenv()

    github_user = os.getenv("GITHUB_USERNAME")
    if github_user is None:
        print("❌ GITHUB_USERNAME missing from .env file.")

    import requests
    import base64

    github_username = github_user
    github_token = os.getenv('GITHUB_TOKEN')
    github_repo = os.getenv('GITHUB_REPO')
    print(f"🔐 Loaded GITHUB_USERNAME={github_username}")
    print(f"🔐 Loaded GITHUB_TOKEN={str(github_token)[:4]+'********' if github_token else None}")
    print(f"🔐 Loaded GITHUB_REPO={github_repo}")

    if not github_token:
        print("❌ GITHUB_TOKEN missing from .env file.")
    if not github_username:
        print("❌ GITHUB_USERNAME missing from .env file.")
    if not github_repo:
        print("❌ GITHUB_REPO missing from .env file.")
    if not all([github_token, github_username, github_repo]):
        return

    token = github_token
    repo = github_repo

    api_url = f"https://api.github.com/repos/{repo}/contents/{github_path}"
    with open(file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Check if file already exists
    get_resp = requests.get(api_url, headers=headers)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

    data = {
        "message": f"Auto-upload {github_path}",
        "content": content,
        "branch": "main",
    }

    if sha:
        data["sha"] = sha  # update existing file

    response = requests.put(api_url, headers=headers, json=data)
    if response.status_code in [200, 201]:
        print(f"✅ Uploaded to GitHub: {github_path}")
    else:
        print(f"❌ GitHub upload failed for {github_path}: {response.text}")

# 🧠 Final Tip: For a clean push after race generation, run:
# git add *.html value_bets.json value.html index.html
# git commit -m "Update racecards and value files"
# git push -f origin main


import argparse
from github import Github

def upload_file_to_github(repo, path, content, message, token):
    """
    Simple GitHub API helper to create or update a file.
    """
    gh = Github(token)
    gh_repo = gh.get_repo(repo)
    try:
        source = gh_repo.get_contents(path)
        gh_repo.update_file(path, message, content, source.sha)
    except Exception:
        gh_repo.create_file(path, message, content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Scrape & generate HTML locally, or deploy existing HTML to GitHub"
    )
    parser.add_argument(
        "--scrape", action="store_true",
        help="Run the full scraper + model + HTML generator locally"
    )
    parser.add_argument(
        "--deploy", action="store_true",
        help="Push previously-generated HTML files up to GitHub via API"
    )
    parser.add_argument(
        "--repo", type=str, default=os.getenv("GITHUB_REPOSITORY"),
        help="GitHub repo in form 'owner/name'"
    )
    parser.add_argument(
        "--token", type=str, default=os.getenv("GITHUB_TOKEN"),
        help="A PAT (repo:contents) stored in GITHUB_TOKEN"
    )
    args = parser.parse_args()

    # Gather custom weights
    weights = input_custom_weights()

    # Define your race URLs or input CSVs here
    race_urls = [
        "https://www.sportinglife.com/racing/racecards/2025-06-09/pontefract/racecard/861786/racing-for-everyone-handicap-gbbplus-race"
    ]

if args.scrape:
    for url in race_urls:
        csv_file = fetch_race_card_data(url)
        model_race(csv_file, weights)
    print("✅ Scrape & build complete.")


    if args.deploy:
        for fname in os.listdir("."):
            if fname.endswith(".html") and fname not in ("monday1_compliant_full.py",):
                with open(fname, "r", encoding="utf-8") as f:
                    content = f.read()
                upload_file_to_github(
                    args.repo,
                    fname,
                    content,
                    f"Update {fname}",
                    args.token
                )
        print("✅ Deploy complete.")

    if not (args.scrape or args.deploy):
        parser.print_help()
