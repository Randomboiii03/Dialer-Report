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

def plot_unique_calls_by_hour(campaign_data):
    all_hours = pd.DataFrame({'Hour of call_originate_time': range(6, 21)})
    connected_by_hour = campaign_data[campaign_data['system_disposition'] == 'CONNECTED'].drop_duplicates().groupby('Hour of call_originate_time').size().reset_index(name='Connected Calls')
    dialed_by_hour = campaign_data.drop_duplicates().groupby('Hour of call_originate_time').size().reset_index(name='Dialed Calls')

    calls_by_hour = all_hours.merge(connected_by_hour, on='Hour of call_originate_time', how='left').fillna(0)
    calls_by_hour = calls_by_hour.merge(dialed_by_hour, on='Hour of call_originate_time', how='left').fillna(0)

    fig_combined = px.line(calls_by_hour, 
                            x='Hour of call_originate_time', 
                            y=['Connected Calls', 'Dialed Calls'], 
                            title='Unique Connected and Dialed Calls by Hour',
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

def display_disposition_metrics(campaign_data):
    disposition_counts = campaign_data.groupby(['DISPOSITION_2'])['Account'].nunique().reset_index()

    rpc_count = disposition_counts.loc[disposition_counts['DISPOSITION_2'] == 'RPC', 'Account'].values
    rpc_value = rpc_count[0] if rpc_count.size > 0 else 0
    
    ptp_count = disposition_counts.loc[disposition_counts['DISPOSITION_2'].isin(["PTP", "PTP OLD", "PTP NEW", "PTP FF UP"]), 'Account'].values
    ptp_value = ptp_count[0] if ptp_count.size > 0 else 0

    payment_count = disposition_counts.loc[disposition_counts['DISPOSITION_2'] == 'PAYMENT', 'Account'].values
    payment_value = payment_count[0] if payment_count.size > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total No. of RPC", f"{rpc_value:,}")
    col2.metric("Total No. of PTP", f"{ptp_value:,}")
    col3.metric("Total No. of PAYMENT", f"{payment_value:,}")
def display_disposition_metrics_manual(campaign_data):
    """
    Displays disposition metrics for Manual Dial calls.
    """
    # Filter data for Manual Dial
    manual_data = campaign_data[campaign_data['CALL TYPE(Auto/Manual)'] == 'Manual Dial']
    
    # Group by 'DISPOSITION_2' to count unique 'Account's
    disposition_counts_manual = manual_data.groupby(['DISPOSITION_2'])['Account'].nunique().reset_index()
    
    # Extract counts for specific dispositions
    rpc_count_manual = disposition_counts_manual.loc[disposition_counts_manual['DISPOSITION_2'] == 'RPC', 'Account'].values
    rpc_value_manual = rpc_count_manual[0] if len(rpc_count_manual) > 0 else 0
    
    ptp_count_manual = disposition_counts_manual.loc[disposition_counts_manual['DISPOSITION_2'].isin(["PTP", "PTP OLD", "PTP NEW", "PTP FF UP"]), 'Account'].values
    ptp_value_manual = ptp_count_manual[0] if len(ptp_count_manual) > 0 else 0
    
    payment_count_manual = disposition_counts_manual.loc[disposition_counts_manual['DISPOSITION_2'] == 'PAYMENT', 'Account'].values
    payment_value_manual = payment_count_manual[0] if len(payment_count_manual) > 0 else 0
    
    # Display metrics in three columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Manual Dial - Total No. of RPC", f"{rpc_value_manual:,}")
    col2.metric("Manual Dial - Total No. of PTP", f"{ptp_value_manual:,}")
    col3.metric("Manual Dial - Total No. of PAYMENT", f"{payment_value_manual:,}")

def display_disposition_metrics_auto(campaign_data):
    """
    Displays disposition metrics for Auto Dial calls.
    """
    # Filter data for Auto Dial
    auto_data = campaign_data[campaign_data['CALL TYPE(Auto/Manual)'] == 'Auto Dial']
    
    # Group by 'DISPOSITION_2' to count unique 'Account's
    disposition_counts_auto = auto_data.groupby(['DISPOSITION_2'])['Account'].nunique().reset_index()
    
    # Extract counts for specific dispositions
    rpc_count_auto = disposition_counts_auto.loc[disposition_counts_auto['DISPOSITION_2'] == 'RPC', 'Account'].values
    rpc_value_auto = rpc_count_auto[0] if len(rpc_count_auto) > 0 else 0
    
    ptp_count_auto = disposition_counts_auto.loc[disposition_counts_auto['DISPOSITION_2'].isin(["PTP", "PTP OLD", "PTP NEW", "PTP FF UP"]), 'Account'].values
    ptp_value_auto = ptp_count_auto[0] if len(ptp_count_auto) > 0 else 0
    
    payment_count_auto = disposition_counts_auto.loc[disposition_counts_auto['DISPOSITION_2'] == 'PAYMENT', 'Account'].values
    payment_value_auto = payment_count_auto[0] if len(payment_count_auto) > 0 else 0
    
    # Display metrics in three columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Auto Dial - Total No. of RPC", f"{rpc_value_auto:,}")
    col2.metric("Auto Dial - Total No. of PTP", f"{ptp_value_auto:,}")
    col3.metric("Auto Dial - Total No. of PAYMENT", f"{payment_value_auto:,}")


# def plot_disposition_distribution(campaign_data):
#     disposition_counts = campaign_data.groupby(['username', 'DISPOSITION_2'])['Account'].nunique().reset_index()
#     total_counts = campaign_data.groupby('username')['Account'].nunique().reset_index().rename(columns={'Account': 'Total_Dialed'})
#     disposition_counts = disposition_counts.merge(total_counts, on='username')
#     disposition_counts['Percentage'] = (disposition_counts['Account'] / disposition_counts['Total_Dialed'] * 100).round().astype(int)

#     unique_dispositions = disposition_counts['DISPOSITION_2'].unique()
#     options = list(unique_dispositions)
#     default_options = [item for item in options if item in ["PTP", "PTP OLD", "PTP NEW", "PTP FF UP", "RPC"]]

#     selected_dispos = st.multiselect('Select Dispositions to Show:', options=options, default=default_options)

#     display_disposition_metrics(campaign_data)

#     fig2 = go.Figure()
#     colors = plotly.colors.qualitative.Vivid

#     if selected_dispos:
#         for i, dispo in enumerate(selected_dispos):
#             color = colors[i % len(colors)]
#             subset = disposition_counts[disposition_counts['DISPOSITION_2'] == dispo]
#             fig2.add_trace(go.Bar(
#                 y=subset['username'],
#                 x=subset['Account'],
#                 name=dispo,
#                 orientation='h',
#                 marker=dict(color=color),
#                 text=[f"{count:,} ({percent}%)" for count, percent in zip(subset['Account'], subset['Percentage'])],
#                 textposition='outside'
#             ))

#     fig2.update_layout(
#         title='Disposition Distribution per Agent (Unique Accounts)',
#         xaxis_title='Number of Unique Accounts',
#         yaxis_title='Agents',
#         barmode='stack', 
#         height=max(500, len(disposition_counts['username'].unique()) * 50),
#         width=10000
#     )

#     st.plotly_chart(fig2)

def plot_disposition_distribution(campaign_data):
    # Group by 'username' and 'DISPOSITION_2' to count unique 'Account's
    campaign_data['DISPOSITION_2'] = campaign_data['DISPOSITION_2'].replace('OTHERS', 'SYSTEM DISPOSITION')
    disposition_counts = campaign_data.groupby(['username', 'DISPOSITION_2'])['Account'].nunique().reset_index()
    
    # Calculate total unique 'Account's per 'username'
    total_counts = campaign_data.groupby('username')['Account'].nunique().reset_index().rename(columns={'Account': 'Total_Dialed'})
    
    # Merge the disposition counts with total counts
    disposition_counts = disposition_counts.merge(total_counts, on='username')
    
    # Calculate the percentage of each disposition per user
    disposition_counts['Percentage'] = (disposition_counts['Account'] / disposition_counts['Total_Dialed'] * 100).round().astype(int)
    
    # Sort the DataFrame by 'Total_Dialed' in descending order
    disposition_counts = disposition_counts.sort_values(by='Total_Dialed', ascending=False)
    
    # Optional: If you want to sort usernames alphabetically instead, comment out the above line and uncomment below
    # disposition_counts = disposition_counts.sort_values(by='username')
    
    # Get unique dispositions
    unique_dispositions = disposition_counts['DISPOSITION_2'].unique()
    options = list(unique_dispositions)
    
    # Define default dispositions to display
    default_options = [item for item in options if item in ["PTP", "PTP OLD", "PTP NEW", "PTP FF UP", "RPC"]]
    
    # Multiselect widget for selecting dispositions
    with st.expander("Filter Options", expanded=False):
        selected_dispos = st.multiselect('Select Dispositions to Show:', options=options, default=default_options)
    
    # Display additional metrics (assuming this function is defined elsewhere)
    display_disposition_metrics(campaign_data)
    
    # Initialize Plotly figure
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
    
    # Update layout with sorted usernames
    fig2.update_layout(
        title='Disposition Distribution per Agent (Unique Accounts)',
        xaxis_title='Number of Unique Accounts',
        yaxis_title='Agents',
        barmode='stack', 
        height=max(500, len(disposition_counts['username'].unique()) * 50),
        # Consider adjusting the width to a more reasonable size if needed
        # width=1000
    )
    
    # To ensure the y-axis follows the sorted order, set categoryorder
    fig2.update_yaxes(categoryorder='total ascending')  # 'total ascending' or 'total descending' as needed
    
    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig2)

# def plot_average_talk_time(campaign_data):
#     connected_calls = campaign_data[campaign_data['system_disposition'] == 'CONNECTED'].drop_duplicates()
#     connected_calls['Talk Time in Seconds'] = connected_calls['End Time in Seconds'] - connected_calls['Start Time in Seconds']
#     avg_talk_time = connected_calls.groupby('username')['Talk Time in Seconds'].mean().reset_index()
#     avg_talk_time['Minutes'] = (avg_talk_time['Talk Time in Seconds'] // 60).astype(int)
#     avg_talk_time['Seconds'] = (avg_talk_time['Talk Time in Seconds'] % 60).astype(int)
#     avg_talk_time['Formatted Talk Time'] = avg_talk_time['Minutes'].astype(str) + ' min ' + avg_talk_time['Seconds'].astype(str) + ' sec'

#     fig_talk_time = px.bar(
#         avg_talk_time,
#         y='username',
#         x='Talk Time in Seconds',
#         title='Average Connected Talk Time per Agent',
#         orientation='h',
#         labels={'Talk Time in Seconds': 'Average Talk Time (in Seconds)', 'username': 'Agent'},
#         text='Formatted Talk Time'
#     )

#     fig_talk_time.update_traces(texttemplate='%{text}', textposition='outside')
#     fig_talk_time.update_layout(
#         yaxis_title='Agent',
#         xaxis_title='Average Talk Time (Seconds)',
#         xaxis_tickangle=-45,
#         height=max(500, len(avg_talk_time) * 50),
#         margin=dict(t=100)
#     )

#     st.plotly_chart(fig_talk_time)
def plot_average_talk_time(campaign_data):
    # Filter connected calls and remove duplicates
    connected_calls = campaign_data[campaign_data['system_disposition'] == 'CONNECTED'].drop_duplicates()
    
    # Calculate talk time in seconds
    connected_calls['Talk Time in Seconds'] = connected_calls['End Time in Seconds'] - connected_calls['Start Time in Seconds']
    
    # Calculate average talk time per agent
    avg_talk_time = connected_calls.groupby('username')['Talk Time in Seconds'].mean().reset_index()
    
    # Sort the DataFrame by 'Talk Time in Seconds' in descending order
    avg_talk_time = avg_talk_time.sort_values(by='Talk Time in Seconds', ascending=True)
    
    # Convert talk time to minutes and seconds for display
    avg_talk_time['Minutes'] = (avg_talk_time['Talk Time in Seconds'] // 60).astype(int)
    avg_talk_time['Seconds'] = (avg_talk_time['Talk Time in Seconds'] % 60).astype(int)
    avg_talk_time['Formatted Talk Time'] = avg_talk_time['Minutes'].astype(str) + ' min ' + avg_talk_time['Seconds'].astype(str) + ' sec'

    # Create a horizontal bar chart
    fig_talk_time = px.bar(
        avg_talk_time,
        y='username',
        x='Talk Time in Seconds',
        title='Average Connected Talk Time per Agent',
        orientation='h',
        labels={'Talk Time in Seconds': 'Average Talk Time (in Seconds)', 'username': 'Agent'},
        text='Formatted Talk Time'
    )

    # Update text position and layout
    fig_talk_time.update_traces(texttemplate='%{text}', textposition='outside')
    fig_talk_time.update_layout(
        yaxis_title='Agent',
        xaxis_title='Average Talk Time (Seconds)',
        xaxis_tickangle=-45,
        height=max(500, len(avg_talk_time) * 50),
        margin=dict(t=100)
    )

    # Display the plot in Streamlit
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
def plot_agent_disposition_manual(campaign_data):
    """
    Plots the disposition distribution per agent for Manual Dial calls,
    including percentage labels for each disposition segment.
    """
    # Display metrics for Manual Dial calls
    campaign_data['DISPOSITION_2'] = campaign_data['DISPOSITION_2'].replace('OTHERS', 'SYSTEM DISPOSITION')
    display_disposition_metrics_manual(campaign_data)
    
    # Filter data for Manual Dial
    manual_data = campaign_data[campaign_data['CALL TYPE(Auto/Manual)'] == 'Manual Dial']
    
    if manual_data.empty:
        st.warning("No Manual Dial data available for this campaign.")
        return
    
    # Group by 'username' and 'DISPOSITION_2' to count unique 'Account's
    disposition_counts_manual = manual_data.groupby(['username', 'DISPOSITION_2'])['Account'].nunique().reset_index(name='Count')
    
    # Pivot the data to have dispositions as columns
    disposition_pivot_manual = disposition_counts_manual.pivot_table(
        index='username',
        columns='DISPOSITION_2',
        values='Count',
        fill_value=0
    ).reset_index()
    
    # Calculate total dispositions per agent for percentage calculation
    disposition_pivot_manual['Total'] = disposition_pivot_manual[disposition_pivot_manual.columns.difference(['username'])].sum(axis=1)
    
    # Calculate percentage for each disposition
    for dispo in disposition_pivot_manual.columns:
        if dispo != 'username' and dispo != 'Total':
            disposition_pivot_manual[f'{dispo}_Percent'] = (disposition_pivot_manual[dispo] / disposition_pivot_manual['Total'] * 100).round(1)
    
    # Create a list of dispositions for consistent coloring
    dispositions = manual_data['DISPOSITION_2'].unique().tolist()
    dispositions.sort()  # Sort dispositions for consistent ordering
    
    # Define colors for different dispositions
    colors = px.colors.qualitative.Vivid
    color_map = {dispo: colors[i % len(colors)] for i, dispo in enumerate(dispositions)}
    
    # Create a Plotly figure
    fig_manual = go.Figure()
    
    # Iterate over each disposition to add a bar for each
    for dispo in dispositions:
        fig_manual.add_trace(go.Bar(
            y=disposition_pivot_manual['username'],
            x=disposition_pivot_manual[dispo],
            name=dispo,
            orientation='h',
            marker=dict(color=color_map[dispo]),
            hovertemplate=f'Disposition: {dispo}<br>Agent: %{{y}}<br>Count: %{{x}}<extra></extra>',
            text=disposition_pivot_manual[f'{dispo}_Percent'].astype(str) + '%',
            textposition='inside'
        ))
    
    # Update layout for stacked bars
    fig_manual.update_layout(
        barmode='stack',
        title='Agent Disposition Distribution - Manual Dial',
        xaxis_title='Number of Unique Accounts',
        yaxis_title='Agents',
        legend_title='Disposition',
        height=max(500, len(disposition_pivot_manual['username']) * 50),
        margin=dict(l=150, r=50, t=100, b=50)
    )
    
    # Update y-axis to ensure agents are sorted and fully visible
    fig_manual.update_yaxes(categoryorder='total ascending')
    
    # Update layout to adjust text styling
    fig_manual.update_traces(textfont=dict(color='white', size=10))
    
    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig_manual, use_container_width=True)


def plot_agent_disposition_auto(campaign_data):
    """
    Plots the disposition distribution per agent for Auto Dial calls,
    including percentage labels for each disposition segment.
    """
    # Display metrics for Auto Dial calls
    campaign_data['DISPOSITION_2'] = campaign_data['DISPOSITION_2'].replace('OTHERS', 'SYSTEM DISPOSITION')
    display_disposition_metrics_auto(campaign_data)
    
    # Filter data for Auto Dial
    auto_data = campaign_data[campaign_data['CALL TYPE(Auto/Manual)'] == 'Auto Dial']
    
    if auto_data.empty:
        st.warning("No Auto Dial data available for this campaign.")
        return
    
    # Group by 'username' and 'DISPOSITION_2' to count unique 'Account's
    disposition_counts_auto = auto_data.groupby(['username', 'DISPOSITION_2'])['Account'].nunique().reset_index(name='Count')
    
    # Pivot the data to have dispositions as columns
    disposition_pivot_auto = disposition_counts_auto.pivot_table(
        index='username',
        columns='DISPOSITION_2',
        values='Count',
        fill_value=0
    ).reset_index()
    
    # Calculate total dispositions per agent for percentage calculation
    disposition_pivot_auto['Total'] = disposition_pivot_auto[disposition_pivot_auto.columns.difference(['username'])].sum(axis=1)
    
    # Calculate percentage for each disposition
    for dispo in disposition_pivot_auto.columns:
        if dispo != 'username' and dispo != 'Total':
            disposition_pivot_auto[f'{dispo}_Percent'] = (disposition_pivot_auto[dispo] / disposition_pivot_auto['Total'] * 100).round(1)
    
    # Create a list of dispositions for consistent coloring
    dispositions = auto_data['DISPOSITION_2'].unique().tolist()
    dispositions.sort()  # Sort dispositions for consistent ordering
    
    # Define colors for different dispositions
    colors = px.colors.qualitative.Vivid
    color_map = {dispo: colors[i % len(colors)] for i, dispo in enumerate(dispositions)}
    
    # Create a Plotly figure
    fig_auto = go.Figure()
    
    # Iterate over each disposition to add a bar for each
    for dispo in dispositions:
        fig_auto.add_trace(go.Bar(
            y=disposition_pivot_auto['username'],
            x=disposition_pivot_auto[dispo],
            name=dispo,
            orientation='h',
            marker=dict(color=color_map[dispo]),
            hovertemplate=f'Disposition: {dispo}<br>Agent: %{{y}}<br>Count: %{{x}}<extra></extra>',
            text=disposition_pivot_auto[f'{dispo}_Percent'].astype(str) + '%',
            textposition='inside'
        ))
    
    # Update layout for stacked bars
    fig_auto.update_layout(
        barmode='stack',
        title='Agent Disposition Distribution - Auto Dial',
        xaxis_title='Number of Unique Accounts',
        yaxis_title='Agents',
        legend_title='Disposition',
        height=max(500, len(disposition_pivot_auto['username']) * 50),
        margin=dict(l=150, r=50, t=100, b=50)
    )
    
    # Update y-axis to ensure agents are sorted and fully visible
    fig_auto.update_yaxes(categoryorder='total ascending')
    
    # Update layout to adjust text styling
    fig_auto.update_traces(textfont=dict(color='white', size=10))
    
    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig_auto, use_container_width=True)

def plot_manual_vs_auto_dial(campaign_data):
    """
    Plots a line graph showing the number of unique accounts per disposition per hour,
    separated by Manual Dial and Auto Dial.
    """
    # Define the range of hours
    hours = range(6, 21)  # 6 AM to 8 PM

    # Define the unique dispositions
    dispositions = campaign_data['DISPOSITION_2'].unique().tolist()

    # Define call types
    call_types = ['Manual Dial', 'Auto Dial']

    # Create a DataFrame with all combinations to ensure completeness
    all_combinations = pd.MultiIndex.from_product(
        [hours, call_types, dispositions],
        names=['Hour', 'Call Type', 'Disposition']
    ).to_frame(index=False)

    # Group the data by Hour, Call Type, and Disposition, and count unique 'Account's
    grouped = campaign_data.groupby(
        ['Hour of call_originate_time', 'CALL TYPE(Auto/Manual)', 'DISPOSITION_2']
    )['Account'].unique().reset_index(name='Unique Account Count')

    # Rename columns for consistency
    grouped = grouped.rename(columns={
        'Hour of call_originate_time': 'Hour',
        'CALL TYPE(Auto/Manual)': 'Call Type',
        'DISPOSITION_2': 'Disposition'
    })

    # Merge with all_combinations to ensure all possible combinations are present
    merged = all_combinations.merge(
        grouped,
        on=['Hour', 'Call Type', 'Disposition'],
        how='left'
    ).fillna(0)

    # Convert Unique Account Count to integer
    merged['Unique Account Count'] = merged['Unique Account Count'].astype(int)

    # Create a unique identifier for each line (e.g., Manual Dial - CONNECTED)
    merged['Line Label'] = merged['Call Type'] + ' - ' + merged['Disposition']

    # Define a color palette
    color_palette = px.colors.qualitative.Vivid

    # Create the line graph using Plotly Express
    fig = px.line(
        merged,
        x='Hour',
        y='Unique Account Count',
        color='Line Label',
        markers=True,
        title='Unique Call Dispositions per Hour by Dial Type',
        labels={
            'Hour': 'Hour of Day',
            'Unique Account Count': 'Number of Unique Accounts',
            'Line Label': 'Dial Type & Disposition'
        },
        color_discrete_sequence=color_palette
    )

    # Update layout for better aesthetics
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(title='Number of Unique Accounts'),
        legend_title='Dial Type & Disposition',
        template='plotly_white',
        hovermode='x unified'
    )

    # Add annotations for data points
    fig.update_traces(
        text=merged['Unique Account Count'],
        textposition='top center',
        texttemplate='%{text}'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)




def main():
    uploaded_file = st.sidebar.file_uploader("Choose a XLSX file", type="xlsx")
    
    if uploaded_file is not None:
        df = load_and_decrypt_file(uploaded_file)
        df['Day of call_originate_time'] = df['Day of call_originate_time'].astype(str)
    
        # st.write_stream(note())
        
        campaigns = pd.Series(df['Campaign Name'].unique()).sort_values().tolist()
        selected_campaign = st.sidebar.selectbox('Select Campaign', campaigns)

        try:
            df['Month'] = pd.to_datetime(df['Month'], errors='coerce').dt.strftime('%m-%Y') 
            unique_month = df['Month'].dropna().sort_values().unique()
            selected_month = st.sidebar.selectbox('Select Month', unique_month)
        except:
            selected_month = None
        
        unique_dates = df['Day of call_originate_time'].dropna().sort_values().unique()
        selected_date = st.sidebar.selectbox('Select Date', unique_dates)

        conditions = (df['Campaign Name'] == selected_campaign) & (df['Day of call_originate_time'] == selected_date)

        if selected_month is not None:
            conditions &= (df['Month'] == selected_month)
        
        campaign_data = df[conditions]

        if campaign_data.empty:
            st.subheader(f'No Data')
        else:
            st.subheader(f'Campaign - {selected_campaign}')
            
            display_metrics(campaign_data)
            
            call_cols = st.columns(2)
            with call_cols[0]:
                plot_calls_by_hour(campaign_data)
            with call_cols[1]:
                # plot_connection_rate(campaign_data)
                plot_unique_calls_by_hour(campaign_data)
            plot_connection_rate(campaign_data)
            dispo_cols = st.columns([2, 1])
            with dispo_cols[0]:
                plot_manual_auto(campaign_data)
            with dispo_cols[1]:
                plot_call_type_distribution(campaign_data)
    
            # plot_disposition_distribution(campaign_data)
            plot_average_talk_time(campaign_data)
            st.header("Agent Disposition Distribution by Call Type")
            
            tabs = st.tabs(["All", "Manual Dial", "Auto Dial", "Per/Hr Dispo"])
            with tabs[0]:
                plot_disposition_distribution(campaign_data)
            with tabs[1]:
                plot_agent_disposition_manual(campaign_data)
            with tabs[2]:
                plot_agent_disposition_auto(campaign_data)
            with tabs[3]:
                plot_manual_vs_auto_dial(campaign_data)
            
    
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
