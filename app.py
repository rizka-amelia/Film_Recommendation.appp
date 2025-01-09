import streamlit as st  
import pickle  
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt  
import networkx as nx  

# Pastikan impor plotly dilakukan dengan benar  
try:  
    import plotly.express as px  
    import plotly.graph_objs as go  
    PLOTLY_INSTALLED = True  
except ImportError:  
    PLOTLY_INSTALLED = False  
    st.warning("Plotly tidak terinstal. Beberapa visualisasi tidak akan tersedia.")  

# Load pickle files  
movies = pickle.load(open("movies.pkl", "rb"))  
similarity = pickle.load(open("similarity.pkl", "rb"))  

# Load original data  
df = pd.read_csv("movies (1).csv")  

# Function for movie recommendation  
def recommend(movie, movies, similarity, df):  
    if movie not in movies['Title'].values:  
        return ["Film tidak ditemukan!"]  
    
    # Get the index of the selected movie  
    index = movies[movies['Title'] == movie].index[0]  
    
    # Sort movies by similarity score  
    distances = list(enumerate(similarity[index]))  
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]  
    
    # Return top 5 recommendations with detailed info  
    recommendations = []  
    for i in distances:  
        recommended_index = i[0]  
        original_index = df[df['Title'] == movies.iloc[recommended_index].Title].index[0]  
        recommended_movie = {  
            "Title": df.iloc[original_index].Title,  
            "Rating": df.iloc[original_index].Rating,  
            "Year": df.iloc[original_index].Year,  
            "Certificate": df.iloc[original_index].Certificate,  
            "Runtime": df.iloc[original_index].Runtime,  
            "Directors": df.iloc[original_index].Directors,  
            "Genre": df.iloc[original_index].Genre,
            "Stars": df.iloc[original_index].Stars,  
            "Similarity": round(i[1] * 100, 2)  # Persentase similarity  
        }  
        recommendations.append(recommended_movie)  
    return recommendations  

# Fungsi untuk menampilkan detail film yang dipilih  
def display_selected_movie_details(selected_movie, df):  
    # Cari detail film yang dipilih  
    movie_details = df[df['Title'] == selected_movie].iloc[0]  
    st.subheader(f"{selected_movie}")  
    # Buat kolom untuk tampilan  
    col1, col2 = st.columns(2)  
    
    with col1:  
        st.markdown(f"**Rating:** {movie_details['Rating']} ⭐")  
        st.markdown(f"**Year:** {movie_details['Year']}")
        st.markdown(f"**Certificate:** {movie_details['Certificate']}")    
    with col2:  
        st.markdown(f"**Directors:** {movie_details['Directors']}")  
        st.markdown(f"**Stars:** {movie_details['Stars']}")  
        st.markdown(f"**Genre:** {movie_details['Genre']}")
 
# Styling untuk tabel  
def style_recommendations(recommendations):  
    # Konversi ke DataFrame  
    rec_df = pd.DataFrame(recommendations)  
    
    # Styling dengan Streamlit  
    st.dataframe(  
        rec_df[['Title', 'Rating', 'Year', 'Genre', 'Directors', 'Stars', 'Certificate', 'Similarity']],  
        column_config={  
            "Title": st.column_config.TextColumn(  
                "Title",  
                width="medium",  
            ),  
            "Rating": st.column_config.NumberColumn(  
                "Rating",  
                format="%.1f ⭐",  
            ),  
            "Year": st.column_config.NumberColumn(  
                "Year",  
                format="%d"  
            ),  
            "Genre": st.column_config.TextColumn("Genre"),  
            "Directors": st.column_config.TextColumn("Directors"),  
            "Stars": st.column_config.TextColumn("Stars"),  
            "Certificate": st.column_config.TextColumn("Certificate"),  
            
            "Similarity": st.column_config.ProgressColumn(  
                "Similarity",  
                format="%f%%",  
                min_value=0,  
                max_value=100  
            )  
        },  
        hide_index=True,  
    )  

# Visualisasi Perbandingan Rating  
def visualize_recommendation_metrics(recommendations):  
    rec_df = pd.DataFrame(recommendations)  
    plt.figure(figsize=(10, 6))  
    colors = plt.cm.viridis(np.linspace(0, 1, len(rec_df)))
    plt.barh(rec_df['Title'], rec_df['Rating'], color=colors)  
    plt.xlabel('Rating', fontsize=14)  
    plt.ylabel('Judul Film', fontsize=14)  
    plt.tight_layout()  
    st.pyplot(plt)  

# Distribusi Genre  
def visualize_genre_distribution(recommendations):  
    genre_list = []  
    for genres in recommendations['Genre']:  
        genre_list.extend(genres.split(', '))  
    genre_counts = pd.Series(genre_list).value_counts()  
    plt.figure(figsize=(4, 4))
    genre_counts.plot(
        kind='pie', 
        autopct='%1.1f%%', 
        colors=plt.cm.Paired(np.linspace(0, 1, len(genre_counts))),
        textprops={'fontsize': 6}
    ) 
    plt.ylabel('')  
    st.pyplot(plt)

# Plotly Scatter Plot Interaktif  
def plotly_recommendation_visualization(recommendations, selected_movie):  
    if not PLOTLY_INSTALLED:  
        st.error("Plotly tidak terinstal. Tidak dapat menampilkan visualisasi.")  
        return  
    rec_df = pd.DataFrame(recommendations)  
    fig = px.scatter(  
        rec_df,  
        x='Year',  
        y='Rating',  
        size='Similarity',  
        color='Genre',  
        hover_name='Title',
    )  
    
    # Kustomisasi layout  
    fig.update_layout(  
        xaxis_title='Tahun Rilis',  
        yaxis_title='Rating Film',  
        legend_title='Genre'  
    )  
    st.plotly_chart(fig)  

# Visualisasi Jaringan Kesamaan Film
def create_similarity_network(movies, similarity, selected_movie, recommendations):  
    rec_titles = [rec['Title'] for rec in recommendations]  
    G = nx.Graph()  
    G.add_node(selected_movie, color='red', size=500)  
    for movie in rec_titles:  
        G.add_node(movie, color='blue', size=300)  
        G.add_edge(selected_movie, movie)  
    plt.figure(figsize=(12, 8))  
    pos = nx.spring_layout(G, k=0.5)  
    nx.draw_networkx_nodes(G, pos,   
        node_color=['red' if node == selected_movie else 'skyblue' for node in G.nodes()],  
        node_size=[500 if node == selected_movie else 300 for node in G.nodes()]  
    )  
    nx.draw_networkx_edges(G, pos, alpha=0.5)  
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.axis('off')  
    st.pyplot(plt)

# Konfigurasi halaman Streamlit  
st.set_page_config(  
    page_title="Film Recommender",  
    page_icon="🎬",  
    layout="wide"  
)  

# Judul dan deskripsi  
st.title("🎬 Film Recommendation System")  
st.markdown("""  
    ### Temukan Film Serupa yang Kamu Sukai!  
    Pilih film, dan dapatkan rekomendasi berdasarkan kesamaan.  
""")  

# Dropdown untuk memilih film  
selected_movie = st.selectbox("Pilih Film", movies['Title'].values)  

# State untuk menyimpan rekomendasi  
if 'recommendations' not in st.session_state:  
    st.session_state.recommendations = None  

# State untuk menyimpan rekomendasi  
if 'recommendations' not in st.session_state:  
    st.session_state.recommendations = None  

# Reset rekomendasi jika film baru dipilih  
if 'last_selected_movie' not in st.session_state:  
    st.session_state.last_selected_movie = None  

if selected_movie != st.session_state.last_selected_movie:  
    st.session_state.recommendations = None  # Reset rekomendasi  
    st.session_state.last_selected_movie = selected_movie  # Update film terakhir yang dipilih
    
# Tombol rekomendasi  
if st.button("Dapatkan Rekomendasi", type="primary"):  
    with st.spinner('Mencari rekomendasi...'):  
        # Dapatkan rekomendasi  
        st.session_state.recommendations = recommend(selected_movie, movies, similarity, df)  
        # Tampilkan detail film yang dipilih  
        display_selected_movie_details(selected_movie, df)  
        # Judul section rekomendasi  
        st.subheader(f"Rekomendasi Film Serupa dengan **{selected_movie}**")  
        # Tampilkan rekomendasi dalam tabel  
        style_recommendations(st.session_state.recommendations)     
        st.markdown("---")  

# Section Visualisasi  
if st.session_state.recommendations:  
    st.subheader("Visualisasi Rekomendasi")  
    # Tampilkan semua visualisasi  
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:  
        st.subheader("Perbandingan Rating")  
        visualize_recommendation_metrics(st.session_state.recommendations)  
    with col2:  
        st.subheader("Jaringan Kesamaan Film")  
        create_similarity_network(movies, similarity, selected_movie, st.session_state.recommendations)
    st.markdown("---") 

    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:  
        st.subheader("Scatter Plot Rekomendasi")  
        plotly_recommendation_visualization(  
            st.session_state.recommendations,   
            selected_movie  
        ) 
    with col2:  
        st.subheader("Distribusi Genre")  
        visualize_genre_distribution(pd.DataFrame(st.session_state.recommendations))
        
st.markdown("---")