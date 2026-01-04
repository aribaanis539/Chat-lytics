import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import preprocessor
import helper

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide"
)
st.markdown("""
<style>
/* Increase tab font size */
button[data-baseweb="tab"] {
    font-size: 22px !important;
    padding: 12px 20px !important;
}

/* Increase tab height */
div[data-baseweb="tab-list"] {
    gap: 3rem;
}

/* Active tab styling */
button[data-baseweb="tab"][aria-selected="true"] {
    font-size: 19px !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #ff4b4b !important;
}

/* Tab container spacing */
div[data-testid="stTabs"] {
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ================== SESSION STATE ==================
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #f6f6f6;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("WhatsApp Analyzer")
    st.title("WhatsApp Analyzer")

    uploaded_file = st.file_uploader(
        "Upload WhatsApp Chat (.txt)",
        type=["txt"]
    )

    media_files = st.file_uploader(
        "Upload Media (Images)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_file:
        data = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        df = preprocessor.preprocess(data)

        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')

        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.selectbox(
            "👤 Select User",
            user_list,
            key="selected_user"
        )

        if st.button("Run Analysis"):
            st.session_state.run_analysis = True

# ================== MAIN APP ==================
if uploaded_file and st.session_state.run_analysis:

    # ---------- BASIC STATS ----------
    num_messages, words, num_media_messages, num_links = helper.fetch_stats(
        selected_user, df
    )

    # ================== TABS ==================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Sentiment",
        "Media",
        "Summary"
    ])

    # ================== TAB 1: OVERVIEW ==================
    # ================== TAB 1: OVERVIEW ==================
    with tab1:
        st.subheader("Chat Overview")

        # ===================== TOP STATS =====================
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(
            selected_user, df
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Messages", num_messages)
        c2.metric("Words", words)
        c3.metric("Media", num_media_messages)
        c4.metric("Links", num_links)

        # ===================== TIMELINES =====================
        with st.expander("Message Timelines"):
            st.markdown("**Monthly Timeline**")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'])
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.markdown("**Daily Timeline**")
            daily = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily['only_date'], daily['message'])
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # ===================== ACTIVITY ANALYSIS =====================
        with st.expander("Activity Analysis"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Most Busy Day**")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                st.markdown("**Most Busy Month**")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            st.markdown("**Weekly Activity Heatmap**")
            heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(heatmap, ax=ax)
            st.pyplot(fig)

        # ===================== BUSY USERS =====================
        if selected_user == "Overall":
            with st.expander("Most Busy Users"):
                x, new_df = helper.most_busy_users(df)

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                with col2:
                    st.dataframe(new_df, use_container_width=True)

        # ===================== WORD ANALYSIS =====================
        with st.expander("Word Analysis"):
            st.markdown("**WordCloud**")
            wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

            st.markdown("**Most Common Words**")
            common_words = helper.most_common_words(selected_user, df)
            fig, ax = plt.subplots()
            ax.barh(common_words[0], common_words[1])
            st.pyplot(fig)

        # ===================== EMOJI ANALYSIS =====================
        with st.expander("Emoji Analysis"):
            emoji_df = helper.emoji_helper(selected_user, df)

            if not emoji_df.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(emoji_df, use_container_width=True)

                with col2:
                    fig, ax = plt.subplots()
                    ax.pie(
                        emoji_df[1].head(),
                        labels=emoji_df[0].head(),
                        autopct="%0.2f%%"
                    )
                    st.pyplot(fig)
            else:
                st.info("No emojis found.")
        with st.expander("Response Time Analysis"):
            response_times = helper.response_time_analysis(df)

            if response_times:
                rt_df = (
                    pd.DataFrame.from_dict(
                        response_times,
                        orient='index',
                        columns=['Avg Response Time (min)']
                    )
                    .sort_values(by='Avg Response Time (min)')
                    .reset_index()
                    .rename(columns={'index': 'User'})
                )

                col1, col2 = st.columns([2, 3])

                with col1:
                    st.dataframe(rt_df, use_container_width=True)

                with col2:
                    fig, ax = plt.subplots()
                    ax.barh(
                        rt_df['User'],
                        rt_df['Avg Response Time (min)']
                    )
                    ax.set_xlabel("Minutes")
                    ax.set_title("Average Response Time")
                    st.pyplot(fig)
            else:
                st.info("Not enough data to compute response times.")

    # ================== TAB 2: SENTIMENT ==================
    with tab2:
        st.subheader("😊 Sentiment Insights")

        if 'sentiment' not in df.columns:
            df = helper.add_sentiment(df)

        col_pos, col_neu, col_neg = st.columns(3)

        with col_pos:
            st.markdown("### Positive")
            st.dataframe(
                helper.most_common_messages_by_sentiment(
                    selected_user, df, "Positive", 5
                ),
                use_container_width=True
            )

        with col_neu:
            st.markdown("### Neutral")
            st.dataframe(
                helper.most_common_messages_by_sentiment(
                    selected_user, df, "Neutral", 5
                ),
                use_container_width=True
            )

        with col_neg:
            st.markdown("### Negative")
            st.dataframe(
                helper.most_common_messages_by_sentiment(
                    selected_user, df, "Negative", 5
                ),
                use_container_width=True
            )

    # ================== TAB 3: MEDIA ==================
    with tab3:
        st.subheader("Media Analysis")

        media_count = helper.media_stats(selected_user, df)

        col1, col2 = st.columns([1, 2])

        with col1:
            fig, ax = plt.subplots()
            ax.pie(
                media_count.values,
                labels=media_count.index,
                autopct="%0.1f%%"
            )
            st.pyplot(fig)

        with col2:
            st.markdown("### 🖼 Image Gallery")

            image_files = []
            if media_files:
                for f in media_files:
                    if f.name.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.webp')
                    ):
                        image_files.append(f)

            if image_files:
                cols = 4
                for i in range(0, len(image_files), cols):
                    grid = st.columns(cols)
                    for col, img in zip(grid, image_files[i:i + cols]):
                        col.image(img, use_container_width=True)
            else:
                st.info("No images uploaded.")

    # ================== TAB 4: SUMMARY ==================
    with tab4:
        st.subheader("Chat Summary")

        if 'sentiment' not in df.columns:
            df = helper.add_sentiment(df)

        summary = helper.generate_chat_summary(df)

        st.info(helper.generate_natural_language_summary(summary))

        st.markdown("### Key Insights")
        for point in helper.generate_bullet_summary(summary):
            st.markdown(f"- {point}")

        st.divider()
        st.subheader("Download Report")

        # Ensure data exists
        if 'sentiment' not in df.columns:
            df = helper.add_sentiment(df)

        summary = helper.generate_chat_summary(df)
        sentiment_counts = helper.sentiment_stats(selected_user, df)
        response_times = helper.response_time_analysis(df)

        pdf_bytes = helper.generate_pdf_report(
            summary,
            response_times,
            sentiment_counts
        )

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name="whatsapp_chat_analysis.pdf",
            mime="application/pdf"
        )


else:
    st.info("👈 Upload a WhatsApp chat file and click **Run Analysis**")


