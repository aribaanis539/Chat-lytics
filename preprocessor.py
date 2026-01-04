import re
import pandas as pd


def preprocess(data: str) -> pd.DataFrame:
    lines = data.splitlines()

    # Supports:
    # 5/10/25, 13:09 -
    # 11/1/25, 1:49 PM -
    # 11/1/2025, 1:49 PM -
    pattern = re.compile(
        r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s'
        r'(\d{1,2}:\d{2}(?:[\u202f\s]?[APap][Mm])?)\s-\s'
        r'(.*)$'
    )

    # ---------- Merge multiline messages ----------
    merged_lines = []
    for line in lines:
        if pattern.match(line):
            merged_lines.append(line)
        else:
            if merged_lines:
                merged_lines[-1] += " " + line.strip()

    dates, times, users, messages = [], [], [], []

    for line in merged_lines:
        match = pattern.match(line)
        if not match:
            continue

        date = match.group(1)
        time = match.group(2).replace('\u202f', ' ').strip().upper()
        content = match.group(3)

        if ':' in content:
            user, message = content.split(':', 1)
            users.append(user.strip())
            messages.append(message.strip())
        else:
            users.append('group_notification')
            messages.append(content.strip())

        dates.append(date)
        times.append(time)

    df = pd.DataFrame({
        'date': dates,
        'time': times,
        'user': users,
        'message': messages
    })

    # ---------- Datetime parsing ----------
    df['datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        dayfirst=True,
        errors='coerce'
    )

    # Drop invalid rows (very rare but safe)
    df.dropna(subset=['datetime'], inplace=True)

    # ---------- Date features ----------
    df['only_date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month_num'] = df['datetime'].dt.month
    df['month'] = df['datetime'].dt.month_name()
    df['day'] = df['datetime'].dt.day
    df['day_name'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute

    # ---------- Period (hour buckets) ----------
    df['period'] = df['hour'].apply(
        lambda h: f"{h:02d}-{(h + 1) % 24:02d}"
    )

    # ---------- Media detection ----------
    def detect_media_type(message: str) -> str:
        msg = message.lower()

        if msg == "<media omitted>":
            return "Media"

        if any(ext in msg for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            return "Image"
        if any(ext in msg for ext in ['.mp4', '.avi', '.mov', '.mkv']):
            return "Video"
        if any(ext in msg for ext in ['.mp3', '.ogg', '.wav', '.m4a']):
            return "Audio"
        if any(ext in msg for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls']):
            return "Document"

        return "Text"

    df['media_type'] = df['message'].apply(detect_media_type)

    return df
