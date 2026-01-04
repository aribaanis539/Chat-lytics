import pandas as pd
from collections import Counter
import emoji

from urlextract import URLExtract
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =====================================================
# BASIC UTILITIES
# =====================================================

extract = URLExtract()
analyzer = SentimentIntensityAnalyzer()

# =====================================================
# BASIC STATS
# =====================================================

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = sum(len(msg.split()) for msg in df['message'])
    num_media_messages = df[df['message'] == '<Media omitted>'].shape[0]

    links = []
    for msg in df['message']:
        links.extend(extract.find_urls(msg))

    return num_messages, words, num_media_messages, len(links)


def most_busy_users(df):
    count = df['user'].value_counts().head()
    percent = (
        df['user'].value_counts(normalize=True) * 100
    ).round(2).reset_index()
    percent.columns = ['name', 'percent']
    return count, percent

# =====================================================
# WORD ANALYSIS
# =====================================================

def create_wordcloud(selected_user, df):
    with open('stop_hinglish.txt', encoding='utf-8') as f:
        stop_words = set(f.read().split())

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[
        (df['user'] != 'group_notification') &
        (df['message'] != '<Media omitted>')
    ]

    def clean(msg):
        return " ".join(w for w in msg.lower().split() if w not in stop_words)

    text = temp['message'].apply(clean).str.cat(sep=" ")

    wc = WordCloud(width=500, height=500, background_color='white')
    return wc.generate(text)


def most_common_words(selected_user, df):
    with open('stop_hinglish.txt', encoding='utf-8') as f:
        stop_words = set(f.read().split())

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['message'] != '<Media omitted>']

    words = []
    for msg in temp['message']:
        for word in msg.lower().split():
            if word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20))

# =====================================================
# EMOJI
# =====================================================

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for msg in df['message']:
        emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])

    return pd.DataFrame(Counter(emojis).most_common())

# =====================================================
# TIMELINES
# =====================================================

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = (
        df.groupby(['year', 'month_num', 'month'])
        .count()['message']
        .reset_index()
    )
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)

# =====================================================
# SENTIMENT
# =====================================================

def get_sentiment(message):
    score = analyzer.polarity_scores(message)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    return 'Neutral'


def add_sentiment(df):
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['message'].apply(get_sentiment)
    return df


def most_common_messages_by_sentiment(selected_user, df, sentiment, top_n=10):
    if 'sentiment' not in df.columns:
        return pd.DataFrame()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['sentiment'] == sentiment) & (df['message'] != '<Media omitted>')]

    return pd.DataFrame(
        Counter(temp['message']).most_common(top_n),
        columns=['message', 'count']
    )

# =====================================================
# MEDIA
# =====================================================

def media_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    return df['media_type'].value_counts()


def most_media_shared_users(df):
    media_df = df[df['media_type'] != 'Text']
    if media_df.empty:
        return pd.Series(dtype=int)
    return media_df['user'].value_counts().head(10)

# =====================================================
# CHAT SUMMARY (NO TOPICS)
# =====================================================

def generate_chat_summary(df):
    summary = {}
    summary['date_range'] = f"{df['only_date'].min()} to {df['only_date'].max()}"
    summary['total_messages'] = df.shape[0]
    summary['total_users'] = df['user'].nunique()
    summary['most_active_user'] = df['user'].value_counts().idxmax()

    summary['dominant_sentiment'] = (
        df['sentiment'].value_counts().idxmax()
        if 'sentiment' in df.columns else "Not computed"
    )

    peak = df['hour'].value_counts().idxmax()
    summary['peak_hour'] = f"{peak}:00 - {peak+1}:00"

    return summary


def generate_natural_language_summary(summary):
    return (
        f"This WhatsApp chat spans from {summary['date_range']}. "
        f"{summary['total_messages']} messages were exchanged among "
        f"{summary['total_users']} participants. "
        f"The most active user was {summary['most_active_user']}. "
        f"Peak activity occurred between {summary['peak_hour']}."
    )


def generate_bullet_summary(summary):
    return [
        f"📅 Chat duration: {summary['date_range']}",
        f"💬 Total messages: {summary['total_messages']}",
        f"👥 Participants: {summary['total_users']}",
        f"🏆 Most active user: {summary['most_active_user']}",
        f"⏰ Peak hour: {summary['peak_hour']}",
    ]



def response_time_analysis(df):
    """
    Calculates average response time (in minutes) per user
    based on consecutive messages from different users.
    """

    # Sort by time
    df = df.sort_values('datetime').reset_index(drop=True)

    response_times = {}

    for i in range(1, len(df)):
        prev_user = df.loc[i - 1, 'user']
        curr_user = df.loc[i, 'user']

        # Ignore same user replies & system messages
        if (
            prev_user == curr_user or
            curr_user == 'group_notification' or
            prev_user == 'group_notification'
        ):
            continue

        time_diff = (
            df.loc[i, 'datetime'] - df.loc[i - 1, 'datetime']
        ).total_seconds() / 60  # minutes

        # Ignore extremely large gaps (e.g., days)
        if time_diff <= 0 or time_diff > 1440:
            continue

        response_times.setdefault(curr_user, []).append(time_diff)

    # Average response time per user
    avg_response_time = {
        user: round(sum(times) / len(times), 2)
        for user, times in response_times.items()
        if len(times) >= 3  # minimum samples for reliability
    }

    return avg_response_time


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import io


def generate_pdf_report(summary, response_times, sentiment_counts):
    """
    Generates a PDF report and returns it as bytes
    """

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    # ---------- TITLE ----------
    elements.append(Paragraph(
        "<b>WhatsApp Chat Analysis Report</b>",
        styles['Title']
    ))
    elements.append(Spacer(1, 0.3 * inch))

    # ---------- SUMMARY ----------
    elements.append(Paragraph("<b>Chat Summary</b>", styles['Heading2']))
    for key, value in summary.items():
        elements.append(
            Paragraph(f"<b>{key.replace('_',' ').title()}:</b> {value}", styles['Normal'])
        )
    elements.append(Spacer(1, 0.3 * inch))

    # ---------- SENTIMENT ----------
    if not sentiment_counts.empty:
        elements.append(Paragraph("<b>Sentiment Distribution</b>", styles['Heading2']))
        sentiment_table = [["Sentiment", "Messages"]] + [
            [idx, int(val)] for idx, val in sentiment_counts.items()
        ]
        elements.append(Table(sentiment_table))
        elements.append(Spacer(1, 0.3 * inch))

    # ---------- RESPONSE TIME ----------
    if response_times:
        elements.append(Paragraph("<b>Average Response Time (minutes)</b>", styles['Heading2']))
        rt_table = [["User", "Avg Response Time (min)"]] + [
            [user, time] for user, time in response_times.items()
        ]
        elements.append(Table(rt_table))

    doc.build(elements)
    buffer.seek(0)

    return buffer.getvalue()
def sentiment_stats(selected_user, df):
    """
    Returns count of Positive / Neutral / Negative messages
    """
    if 'sentiment' not in df.columns:
        return pd.Series(dtype=int)

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    return df['sentiment'].value_counts()

