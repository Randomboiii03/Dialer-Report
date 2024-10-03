import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors
import io

st.set_page_config(layout="wide")
st.title('Call Analytics Dashboard')

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Sidebar for campaign selection
    campaigns = df['Campaign Name'].unique()
    selected_campaign = st.sidebar.selectbox('Select Campaign', campaigns)
    all_hours = pd.DataFrame({'Hour of call_originate_time': range(6, 21)})
    
    # Filter data for selected campaign
    campaign_data = df[df['Campaign Name'] == selected_campaign]
    st.subheader(f'Campaign - {selected_campaign}')
    total_unique_accounts = campaign_data['Account'].nunique()
    total_connected = campaign_data[campaign_data['system_disposition'] == 'CONNECTED']['Account'].nunique()
    overall_connection_rate = total_connected / total_unique_accounts if total_unique_accounts > 0 else 0

    total_calls = campaign_data['Account'].count()
    penetration_rate = total_calls / total_unique_accounts if total_unique_accounts > 0 else 0

    col1, col2, col3 = st.columns(3)
    col2.metric("Total Dialed", f"{total_calls:,}")
    col1.metric("Total Unique Accounts", f"{total_unique_accounts:,}")
    col3.metric("Total Connected", f"{total_connected:,}")

    col1, col2, col3 = st.columns(3)
    col2.metric("Penetration Rate", f"{penetration_rate:.0%}")
    col3.metric("Overall Connection Rate", f"{overall_connection_rate:.0%}")  # Rounded to nearest percent

    col1, col2 = st.columns(2)
    with col1:
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
                                    texttemplate='%{y:.0f}',  # Display as integer
                                    textposition='top center')

        fig_combined.update_xaxes(range=[6, 20], tickmode='linear', dtick=1)
        fig_combined.update_layout(xaxis_title="Hour of Day",
                                    yaxis_title="Number of Calls",
                                    legend_title="",
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        st.plotly_chart(fig_combined)

    with col2:
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
                        texttemplate='%{y:.0%}',  # Display as rounded percentage
                        textposition='top center')

        fig3.update_xaxes(range=[6, 20], tickmode='linear', dtick=1, title="Hour of Day")
        fig3.update_yaxes(title='Connection Rate', tickformat='.0%')  # Rounded to nearest percent
        fig3.update_layout(legend_title="",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig3)

    # Disposition Distribution
    col1, col2 = st.columns([2, 1])
    with col1:
        manual_auto_connected = campaign_data[campaign_data['system_disposition'] == 'CONNECTED'].groupby(['Hour of call_originate_time', 'CALL TYPE(Auto/Manual)']).size().reset_index(name='Connected Calls')
        manual_auto_connected_pivot = manual_auto_connected.pivot(index='Hour of call_originate_time', 
                                                            columns='CALL TYPE(Auto/Manual)', 
                                                            values='Connected Calls').reset_index()
        manual_auto_connected_pivot = manual_auto_connected_pivot.rename(columns={'Auto': 'Auto Dial', 'Manual': 'Manual Dial'})
        manual_auto_connected_pivot = all_hours.merge(manual_auto_connected_pivot, on='Hour of call_originate_time', how='left').fillna(0)

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
                texttemplate='%{text:.0f}'  # Display as integer
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

    # Call Type Distribution
    with col2:
        call_type_dist = campaign_data['CALL TYPE(Auto/Manual)'].value_counts()
        total_calls = call_type_dist.sum()
        call_type_percentages = (call_type_dist / total_calls * 100).round().astype(int)  # Round to nearest percent

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
            texttemplate='%{label}<br>%{percent:.0%}'  # Display as rounded percentage
        )

        st.plotly_chart(fig4)
    
    # Replace 'OTHERS' with 'SYSTEM DISPOSITION' in DISPOSITION_2
    campaign_data['DISPOSITION_2'] = campaign_data['DISPOSITION_2'].replace('OTHERS', 'SYSTEM DISPOSITION')

    # Group by username and DISPOSITION_2, then count unique accounts
    disposition_counts = campaign_data.groupby(['username', 'DISPOSITION_2'])['Account'].nunique().reset_index()

    # Calculate the total unique accounts dialed per username
    total_counts = campaign_data.groupby('username')['Account'].nunique().reset_index().rename(columns={'Account': 'Total_Dialed'})

    # Merge total accounts with disposition counts
    disposition_counts = disposition_counts.merge(total_counts, on='username')

    # Calculate the percentage for each disposition
    disposition_counts['Percentage'] = (disposition_counts['Account'] / disposition_counts['Total_Dialed'] * 100).round().astype(int)

    # Get unique dispositions for the selection
    unique_dispositions = disposition_counts['DISPOSITION_2'].unique()

    # Add "Total Dialed" to the options
    options = list(unique_dispositions)

    # Streamlit multiselect for selecting dispositions
    selected_dispos = st.multiselect('Select Dispositions to Show:', options=options, default=['PTP NEW', 'PTP OLD', 'PTP FF UP', 'RPC', 'PAYMENT'])

    # Create a bar chart
    fig2 = go.Figure()
    colors = plotly.colors.qualitative.Vivid

    # Check if any dispositions are selected
    if selected_dispos:
        for i, dispo in enumerate(selected_dispos):
            # Get the subset of disposition counts

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
                

    # Update layout for the chart
    fig2.update_layout(
        title='Disposition Distribution per Agent (Unique Accounts)',
        xaxis_title='Number of Unique Accounts',
        yaxis_title='Agents',
        barmode='stack', 
        height=max(500, len(disposition_counts['username'].unique()) * 30),
        width=10000
    )

    st.plotly_chart(fig2)


    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
        }

    system_prompt = f"""
        Analyze the following data and summarize it in a clear and concise manner, using bullet points and categorizing it by campaign make use of different emojis and millenial wordings appropiate on the datas, also shorten the analysis..
        Please ensure that the summary is accurate and easy to understand, and that each campaign is clearly separated by its respective category or label.
        Add the range of hours of call of the data to indicate what existing data we have. 
        Additionally, please use standard formatting and formatting guidelines to ensure the output is clean and readable. 
        Example output: **Campaign 1:** • Metrics: [Insert metrics here]
        """

    # system_prompt = f"""
    #     You are a data and performance analysis
    #     """

    # prompt = f"""
    #     Summarize {selected_campaign} campaign. Here's additional data:
    #     Total Dialed: {total_calls}
    #     Total Unique Dials: {total_unique_accounts}
    #     Penetration Rate: {penetration_rate}
    #     Total Connected: {total_connected}
    #     Overall Connection Rate: {overall_connection_rate}
    #     Calls By Hour: {calls_by_hour}
    #     Call Type By hour: {hourly_stats}
    #     Dispo Details: {}
    #     """

    # model = genai.GenerativeModel(
    #     model_name="gemini-1.5-flash",
    #     generation_config=generation_config,
    #     system_instruction=system_prompt)
    # response = model.generate_content(prompt)
    # st.write(response.text)

else:
    st.write("Please upload a CSV file to begin the analysis.")