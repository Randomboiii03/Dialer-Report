import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors
import io
import msoffcrypto
import google.generativeai as genai
from datetime import datetime
from time import sleep

st.set_page_config(page_title="", layout="wide")
st.title('Call Analytics Dashboard')

genai.configure(api_key=st.secrets["API_KEY"])

@st.cache_data
def load_and_decrypt_file(uploaded_file):
    decrypted = io.BytesIO()
    encrypted = msoffcrypto.OfficeFile(uploaded_file)
    encrypted.load_key(password=st.secrets["PASSWORD"])
    encrypted.decrypt(decrypted)
    decrypted.seek(0)
    return pd.read_excel(decrypted)

def note():
    note = ":red[**Note:**] The data presented here may be incomplete due to the system's heavy computational load."
    for word in note.split(" "):
        yield word + " "
        sleep(0.05)

def display_metrics(campaign_data):
    total_unique_accounts = campaign_data['Account'].nunique()
    total_connected = campaign_data[campaign_data['system_disposition'] == 'CONNECTED']['Account'].nunique()
    overall_connection_rate = total_connected / total_unique_accounts if total_unique_accounts > 0 else 0
    total_calls = campaign_data['Account'].count()
    penetration_rate = total_calls / total_unique_accounts if total_unique_accounts > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Unique Accounts", f"{total_unique_accounts:,}")
    col2.metric("Total Dialed", f"{total_calls:,}")
    col3.metric("Total Connected", f"{total_connected:,}")

    col1, col2, col3 = st.columns(3)
    col2.metric("Penetration Rate", f"{penetration_rate:.0%}")
    col3.metric("Overall Connection Rate", f"{overall_connection_rate:.0%}")

def plot_calls_by_hour(campaign_data):
    all_hours = pd.DataFrame({'Hour of call_originate_time': range(6, 21)})
    connected_by_hour = campaign_data[campaign_data['system_disposition'] == 'CONNECTED'].groupby('Hour of call_originate_time').size().reset_index(name='Connected Calls')
    dialed_by_hour = campaign_data.groupby('Hour of call_originate_time').size().reset_index(name='Dialed Calls')

    calls_by_hour = all_hours.merge(connected_by_hour, on='Hour of call_originate_time', how='left').fillna(0)
    calls_by_hour = calls_by_hour.merge(dialed_by_hour, on='Hour of call_originate_time', how='left').fillna(0)

    fig_combined = px.line(calls_by_hour, 
                            x='Hour of call_originate_time', 
                            y=['Connected Calls', 'Dialed Calls'], 
                            title='Connected and Dialed Calls by Hour',
                            markers=True,
                            labels={'value': 'Number of Calls', 'variable': 'Call Type'})

    fig_combined.update_traces(mode='lines+markers+text',
                                marker=dict(size=8),
                                texttemplate='%{y:.0f}',
                                textposition='top center')
    fig_combined.update_xaxes(range=[6, 20], tickmode='linear', dtick=1)
    fig_combined.update_layout(xaxis_title="Hour of Day",
                                yaxis_title="Number of Calls",
                                legend_title="",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    st.plotly_chart(fig_combined)

def plot_connection_rate(campaign_data):
    all_hours = pd.DataFrame({'Hour of call_originate_time': range(6, 21)})
    hourly_stats = campaign_data.groupby('Hour of call_originate_time').agg({
        'system_disposition': lambda x: (x == 'CONNECTED').sum(),
        'Account': 'nunique'
    }).reset_index()

    hourly_stats = all_hours.merge(hourly_stats, on='Hour of call_originate_time', how='left').fillna(0)
    hourly_stats['Connection Rate'] = hourly_stats['system_disposition'] / hourly_stats['Account'].replace(0, 1)

    fig3 = px.line(hourly_stats, 
                    x='Hour of call_originate_time', 
                    y='Connection Rate', 
                    title='Connection Rate by Hour',
                    markers=True,
                    line_shape='linear')

    fig3.update_traces(mode='lines+markers+text',
                        marker=dict(size=8),
                        texttemplate='%{y:.0%}', 
                        textposition='top center')

    fig3.update_xaxes(range=[6, 20], tickmode='linear', dtick=1, title="Hour of Day")
    fig3.update_yaxes(title='Connection Rate', tickformat='.0%')
    fig3.update_layout(legend_title="",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig3)

def plot_manual_auto(campaign_data):
    all_hours = pd.DataFrame({'Hour of call_originate_time': range(6, 21)})
    manual_auto_connected = campaign_data[campaign_data['system_disposition'] == 'CONNECTED'].groupby(
        ['Hour of call_originate_time', 'CALL TYPE(Auto/Manual)']
    ).size().reset_index(name='Connected Calls')

    manual_auto_connected_pivot = manual_auto_connected.pivot(
        index='Hour of call_originate_time', 
        columns='CALL TYPE(Auto/Manual)', 
        values='Connected Calls'
    ).reset_index()

    manual_auto_connected_pivot = manual_auto_connected_pivot.rename(columns={'Auto': 'Auto Dial', 'Manual': 'Manual Dial'})
    manual_auto_connected_pivot = all_hours.merge(
        manual_auto_connected_pivot,
        on='Hour of call_originate_time',
        how='left'
    ).fillna(0)

    for call_type in ['Manual Dial', 'Auto Dial']:
        if call_type not in manual_auto_connected_pivot:
            manual_auto_connected_pivot[call_type] = 0

    fig_manual_auto = go.Figure()

    for call_type in ['Manual Dial', 'Auto Dial']:
        fig_manual_auto.add_trace(go.Scatter(
            x=manual_auto_connected_pivot['Hour of call_originate_time'],
            y=manual_auto_connected_pivot[call_type],
            mode='lines+markers+text',
            name=call_type,
            line=dict(width=2),
            marker=dict(size=8),
            text=manual_auto_connected_pivot[call_type],
            textposition='top center',
            texttemplate='%{text:.0f}'
        ))

    fig_manual_auto.update_layout(
        title='Connected Calls by Hour: Manual vs Auto Dial',
        xaxis_title='Hour of Day',
        yaxis_title='Number of Connected Calls',
        legend_title='Call Type',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig_manual_auto.update_xaxes(range=[6, 20], tickmode='linear', dtick=1)
    st.plotly_chart(fig_manual_auto)

def plot_call_type_distribution(campaign_data):
    call_type_dist = campaign_data['CALL TYPE(Auto/Manual)'].value_counts()
    total_calls = call_type_dist.sum()
    call_type_percentages = (call_type_dist / total_calls * 100).round().astype(int)

    fig4 = px.pie(
        values=call_type_dist.values,
        names=call_type_dist.index,
        title='Call Type Distribution',
        labels={'label': 'Call Type', 'value': 'Number of Calls'}
    )

    fig4.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        texttemplate='%{label}<br>%{percent:.0%}'
    )

    st.plotly_chart(fig4)

def plot_disposition_distribution(campaign_data):
    disposition_counts = campaign_data.groupby(['username', 'DISPOSITION_2'])['Account'].nunique().reset_index()
    total_counts = campaign_data.groupby('username')['Account'].nunique().reset_index().rename(columns={'Account': 'Total_Dialed'})
    disposition_counts = disposition_counts.merge(total_counts, on='username')
    disposition_counts['Percentage'] = (disposition_counts['Account'] / disposition_counts['Total_Dialed'] * 100).round().astype(int)

    unique_dispositions = disposition_counts['DISPOSITION_2'].unique()
    options = list(unique_dispositions)
    default_options = [item for item in options if item in ["PTP", "PTP OLD", "PTP NEW", "PTP FF UP", "RPC"]]

    selected_dispos = st.multiselect('Select Dispositions to Show:', options=options, default=default_options)

    fig2 = go.Figure()
    colors = plotly.colors.qualitative.Vivid

    if selected_dispos:
        for i, dispo in enumerate(selected_dispos):
            color = colors[i % len(colors)]
            subset = disposition_counts[disposition_counts['DISPOSITION_2'] == dispo]
            fig2.add_trace(go.Bar(
                y=subset['username'],
                x=subset['Account'],
                name=dispo,
                orientation='h',
                marker=dict(color=color),
                text=[f"{count:,} ({percent}%)" for count, percent in zip(subset['Account'], subset['Percentage'])],
                textposition='outside'
            ))

    fig2.update_layout(
        title='Disposition Distribution per Agent (Unique Accounts)',
        xaxis_title='Number of Unique Accounts',
        yaxis_title='Agents',
        barmode='stack', 
        height=max(500, len(disposition_counts['username'].unique()) * 50),
        width=10000
    )

    st.plotly_chart(fig2)

def plot_average_talk_time(campaign_data):
    connected_calls = campaign_data[campaign_data['system_disposition'] == 'CONNECTED'].drop_duplicates()
    connected_calls['Talk Time in Seconds'] = connected_calls['End Time in Seconds'] - connected_calls['Start Time in Seconds']
    avg_talk_time = connected_calls.groupby('username')['Talk Time in Seconds'].mean().reset_index()
    avg_talk_time['Minutes'] = (avg_talk_time['Talk Time in Seconds'] // 60).astype(int)
    avg_talk_time['Seconds'] = (avg_talk_time['Talk Time in Seconds'] % 60).astype(int)
    avg_talk_time['Formatted Talk Time'] = avg_talk_time['Minutes'].astype(str) + ' min ' + avg_talk_time['Seconds'].astype(str) + ' sec'

    fig_talk_time = px.bar(
        avg_talk_time,
        y='username',
        x='Talk Time in Seconds',
        title='Average Connected Talk Time per Agent',
        orientation='h',
        labels={'Talk Time in Seconds': 'Average Talk Time (in Seconds)', 'username': 'Agent'},
        text='Formatted Talk Time'
    )

    fig_talk_time.update_traces(texttemplate='%{text}', textposition='outside')
    fig_talk_time.update_layout(
        yaxis_title='Agent',
        xaxis_title='Average Talk Time (Seconds)',
        xaxis_tickangle=-45,
        height=max(500, len(avg_talk_time) * 50),
        margin=dict(t=100)
    )

    st.plotly_chart(fig_talk_time)

@st.cache_data
def generate_summary(campaign_data, selected_campaign, total_calls, total_unique_accounts, penetration_rate, total_connected, overall_connection_rate):
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Optimized professional prompt for generating the summary
    system_prompt = f"""
        Dive into the data and craft a snappy, engaging summary of the {selected_campaign} campaign. Use bullet points to highlight the must-know metrics and insights. Keep it clear and relatable, using a tone that resonates with a millennial audience. Don’t forget to include the range of call hours for context!

        **Campaign Highlights: {selected_campaign}**
        - **Total Calls Dialed:** {total_calls} 📞
        - **Total Unique Accounts:** {total_unique_accounts} 👥
        - **Penetration Rate:** {penetration_rate:.0%} 🚀
        - **Total Connected Calls:** {total_connected} 📈
        - **Overall Connection Rate:** {overall_connection_rate:.0%} 🌟
        - **Call Hours Range:** [e.g., 6 AM to 8 PM] ⏰

        Wrap it up with actionable insights and key takeaways that can inspire future strategies. Let’s make this data pop!
    """
    # st.write(system_prompt)
    prompt = f"""
        Summarize the {selected_campaign} campaign using the following data:
        - Total Calls Dialed: {total_calls}
        - Total Unique Accounts: {total_unique_accounts}
        - Penetration Rate: {penetration_rate}
        - Total Connected Calls: {total_connected}
        - Overall Connection Rate: {overall_connection_rate}
    """

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction=system_prompt
    )
    response = model.generate_content(prompt)
    st.write(response.text)

def main():
    uploaded_file = st.sidebar.file_uploader("Choose a XLSX file", type="xlsx")
    
    if uploaded_file is not None:
        df = load_and_decrypt_file(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date.astype(str) 
        df['Day of call_originate_time'] = df['Day of call_originate_time'].astype(str)
        # df['Date'] = pd.to_datetime(df['Day of call_originate_time'])  # Ensure the new Date column is in datetime format
    
        st.write_stream(note())
        
        campaigns = pd.Series(df['Campaign Name'].unique()).sort_values().tolist()

        # current_month = datetime.now().strftime('%B %Y')
        # days = pd.Series(df['Day of call_originate_time'].unique()).sort_values().tolist()
        # days = [f"{day} {current_month}" for day in days]

        selected_campaign = st.sidebar.selectbox('Select Campaign', campaigns)
        # selected_date = st.sidebar.selectbox('Select Date', sorted(unique_dates)) 
        # selected_day = st.sidebar.selectbox('Select Day', days)

        # selected_day_number = selected_day.split(" ")[0]
        # campaign_data = df[
        #     (df['Campaign Name'] == selected_campaign) &
        #     (df['Day of call_originate_time'].str.contains(selected_day_number))
        # ]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        unique_dates = pd.Series(df['Date'].unique()).sort_values().tolist()  # Assuming 'Date' is the new column
        selected_date = st.sidebar.selectbox('Select Date', df['Date'])

        campaign_data = df[
            (df['Campaign Name'] == selected_campaign) &
            (df['Date'] == selected_date)  # Filter by the selected date
        ]

        # Ensure that Day of call_originate_time is filtered to include all hours for the selected date
        campaign_data['Day of call_originate_time'] = campaign_data['Day of call_originate_time'].astype(str)

        st.subheader(f'Campaign - {selected_campaign}')

        display_metrics(campaign_data)
        
        call_cols = st.columns(2)
        with call_cols[0]:
            plot_calls_by_hour(campaign_data)
        with call_cols[1]:
            plot_connection_rate(campaign_data)

        dispo_cols = st.columns([2, 1])
        with dispo_cols[0]:
            plot_manual_auto(campaign_data)
        with dispo_cols[1]:
            plot_call_type_distribution(campaign_data)

        plot_disposition_distribution(campaign_data)
        plot_average_talk_time(campaign_data)

        total_calls = campaign_data['Account'].count()
        total_unique_accounts = campaign_data['Account'].nunique()
        penetration_rate = total_calls / total_unique_accounts if total_unique_accounts > 0 else 0
        total_connected = campaign_data[campaign_data['system_disposition'] == 'CONNECTED']['Account'].nunique()
        overall_connection_rate = total_connected / total_unique_accounts if total_unique_accounts > 0 else 0

        generate_summary(campaign_data, selected_campaign, total_calls, total_unique_accounts, penetration_rate, total_connected, overall_connection_rate)
    else:
        st.write("Please upload a XLSX file to begin the analysis.")

if __name__ == "__main__":
    main()
