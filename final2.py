import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="Argo Float Profiles Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        color: #000000;    
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_data():
    """Generate enhanced sample data for Argo float profiles"""
    np.random.seed(42)
    floats_data = []
    
    for float_id in range(1, 6):
        base_lat = 30 + np.random.uniform(-10, 10)
        base_lon = -120 + np.random.uniform(-20, 20)
        
        for profile in range(3):  # 3 profiles per float
            date = datetime.now() - timedelta(days=profile * 30)
            depths = np.arange(0, 1000, 50)
            
            for depth in depths:
                # Realistic temperature profile (decreases with depth)
                temp = 25 * np.exp(-depth/500) + 2 + np.random.normal(0, 0.5)
                # Realistic salinity profile
                salinity = 34 + 1.5 * (1 - np.exp(-depth/200)) + np.random.normal(0, 0.1)
                # Pressure approximates depth
                pressure = depth * 1.025 + np.random.normal(0, 2)
                
                floats_data.append({
                    'float_id': float_id,
                    'profile_id': f"{float_id}_{profile}",
                    'depth': depth,
                    'temp': max(0, temp),
                    'salinity': max(30, salinity),
                    'pressure': max(0, pressure),
                    'lat': base_lat + np.random.normal(0, 0.1),
                    'lon': base_lon + np.random.normal(0, 0.1),
                    'date': date.strftime('%Y-%m-%d')
                })
    
    return pd.DataFrame(floats_data)

def create_map(floats_df):
    """Create an interactive map showing float locations"""
    # Get latest position for each float
    latest_positions = floats_df.groupby('float_id').agg({
        'lat': 'first',
        'lon': 'first',
        'date': 'max'
    }).reset_index()
    
    # Create base map centered on the data
    center_lat = latest_positions['lat'].mean()
    center_lon = latest_positions['lon'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles="OpenStreetMap"
    )
    
    # Add float markers
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for idx, row in latest_positions.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"Float {row['float_id']}<br>Last Update: {row['date']}",
            tooltip=f"Float {row['float_id']}",
            icon=folium.Icon(color=colors[int(row['float_id']-1)], icon='tint')
        ).add_to(m)
    
    return m

def analyze_data_query(query, df):
    """Simple data analysis based on user query"""
    query_lower = query.lower()
    
    if 'temperature' in query_lower:
        temp_stats = df.groupby('float_id')['temp'].agg(['mean', 'min', 'max'])
        return f"🌡 *Temperature Analysis:*\n\nAverage temperature across floats: {temp_stats['mean'].min():.1f}°C to {temp_stats['mean'].max():.1f}°C\n\nOverall range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C\n\nTemperature decreases with depth due to reduced solar heating."
    
    elif 'salinity' in query_lower:
        sal_stats = df.groupby('float_id')['salinity'].agg(['mean', 'min', 'max'])
        return f"🧂 *Salinity Analysis:*\n\nAverage salinity ranges from {sal_stats['mean'].min():.1f} to {sal_stats['mean'].max():.1f} PSU\n\nSalinity typically increases with depth due to less mixing and higher pressure."
    
    elif 'depth' in query_lower or 'deep' in query_lower:
        max_depth = df['depth'].max()
        avg_measurements = len(df) / len(df['profile_id'].unique())
        return f"🌊 *Depth Analysis:*\n\nMaximum depth recorded: {max_depth:.0f}m\n\nAverage measurements per profile: {avg_measurements:.0f}\n\nOcean properties change significantly with depth due to pressure and light penetration."
    
    elif 'profile' in query_lower:
        return "📊 *Profile Analysis:*\n\nYou can analyze different profiles:\n\n• Switch between different floats\n• Select different profiles for each float\n• Examine temperature and salinity depth profiles\n• View float locations on the map"
    
    elif any(f'float {i}' in query_lower for i in range(1, 6)):
        float_num = next(i for i in range(1, 6) if f'float {i}' in query_lower)
        float_data = df[df['float_id'] == float_num]
        num_profiles = len(float_data['profile_id'].unique())
        location = float_data.iloc[0]
        return f"🎯 *Float {float_num} Information:*\n\nProfiles recorded: {num_profiles}\n\nLocation: {location['lat']:.2f}°N, {abs(location['lon']):.2f}°W\n\nDepth range: {float_data['depth'].min():.0f}m to {float_data['depth'].max():.0f}m\n\nLatest measurement: {float_data['date'].iloc[0]}"
    
    else:
        return "🤖 *How I can help:\n\nI can analyze the oceanographic data! Try asking about:\n\n• **Temperature* patterns and ranges\n• *Salinity* variations with depth\n• *Depth* measurements and coverage\n• Specific *float* information\n• *Profile* analysis and trends\n\nExample: 'What is the temperature range at 200m depth?'"

# Load data
df = generate_sample_data()

# Get unique floats for map
floats = df.groupby('float_id').agg({
    'lat': 'first',
    'lon': 'first',
    'date': 'max'
}).reset_index()

# Main title
st.markdown('<h1 class="main-header">🌊 Argo Float Profiles Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("🎛 Controls")

# Float selection
selected_float = st.sidebar.selectbox("Select Float:", sorted(df['float_id'].unique()))

# Profile selection for selected float
available_profiles = df[df['float_id'] == selected_float]['profile_id'].unique()
profile_labels = {p: f"Profile {p.split('_')[1]}" for p in available_profiles}
selected_profile = st.sidebar.selectbox("Select Profile:", available_profiles, format_func=lambda x: profile_labels[x])

# Clear any session state that might contain comparison data
if 'compare_floats' in st.session_state:
    del st.session_state.compare_floats

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Float Locations")
    
    # Create and display map
    map_obj = create_map(df)
    map_data = st_folium(map_obj, height=400, width=700)

with col2:
    st.subheader("ℹ Float Information")
    
    # Display float information
    if selected_profile:
        profile_data = df[df['profile_id'] == selected_profile].iloc[0]
        num_measurements = len(df[df['profile_id'] == selected_profile])
        max_depth = df[df['profile_id'] == selected_profile]['depth'].max()
        
        st.markdown(f"""
        <div class="info-box">
            <strong>Float ID:</strong> {selected_float}<br>
            <strong>Profile Date:</strong> {profile_data['date']}<br>
            <strong>Location:</strong> {profile_data['lat']:.2f}°N, {abs(profile_data['lon']):.2f}°W<br>
            <strong>Max Depth:</strong> {max_depth:.0f}m<br>
            <strong>Measurements:</strong> {num_measurements}
        </div>
        """, unsafe_allow_html=True)
    
    # Display key metrics for selected profile only
    if selected_profile:
        st.subheader("📊 Profile Metrics")
        
        profile_subset = df[df['profile_id'] == selected_profile]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Avg Temperature", f"{profile_subset['temp'].mean():.1f}°C")
            st.metric("Max Depth", f"{profile_subset['depth'].max():.0f}m")
        with col_b:
            st.metric("Avg Salinity", f"{profile_subset['salinity'].mean():.1f} PSU")
            st.metric("Measurements", f"{len(profile_subset)}")

# Profile visualization - Two graphs side by side
st.subheader("📈 Ocean Profile Analysis")

if selected_profile:
    profile_data = df[df['profile_id'] == selected_profile].sort_values('depth')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature vs Depth graph
        fig_temp = px.line(
            profile_data,
            x='temp',
            y='depth',
            title=f"Temperature Profile - Float {selected_float}",
            markers=True
        )
        
        fig_temp.update_layout(
            yaxis_autorange='reversed',
            xaxis_title='Temperature (°C)',
            yaxis_title='Depth (m)',
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # Salinity vs Depth graph
        fig_sal = px.line(
            profile_data,
            x='salinity',
            y='depth',
            title=f"Salinity Profile - Float {selected_float}",
            markers=True,
            color_discrete_sequence=['#ff7f0e']  # Different color for salinity
        )
        
        fig_sal.update_layout(
            yaxis_autorange='reversed',
            xaxis_title='Salinity (PSU)',
            yaxis_title='Depth (m)',
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_sal, use_container_width=True)
else:
    st.info("Select a profile to display the graphs")



# Data Assistant (Chatbot)
st.subheader("🤖 Data Assistant")
st.write("Ask questions about the oceanographic data:")

# Chat interface
chat_input = st.text_area(
    "Your question:",
    placeholder='e.g., "What is the temperature range at 200m depth?" or "Compare salinity between floats 1 and 2"',
    height=80
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🚀 Ask", type="primary"):
        if chat_input:
            # Initialize chat history if not exists
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Get response
            response = analyze_data_query(chat_input, df)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'user': chat_input,
                'assistant': response,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })

with col2:
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []

# Display chat history
if 'chat_history' in st.session_state and st.session_state.chat_history:
    st.subheader("💬 Conversation History")
    
    # Show latest conversations first
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 exchanges
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You ({chat['timestamp']}):</strong><br>
            {chat['user']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong><br>
            {chat['assistant']}
        </div>
        """, unsafe_allow_html=True)
        
        if i < len(st.session_state.chat_history[-5:]) - 1:
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        🌊 Argo Float Profiles Dashboard | Built with Streamlit & Plotly<br>
        <small>Synthetic oceanographic data for demonstration purposes</small>
    </div>
    """,
    unsafe_allow_html=True
)