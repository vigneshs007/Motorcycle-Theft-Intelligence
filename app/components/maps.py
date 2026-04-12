import streamlit as st
import pandas as pd
import os

def show_route_map(df_points: pd.DataFrame, lat_col="lat", lon_col="lon", zoom_start=13):
    """
    Display a route map. Uses folium + streamlit_folium if available, otherwise falls back to st.map.
    Expects df_points with numeric latitude/longitude columns.
    """
    # basic validation
    if lat_col not in df_points.columns or lon_col not in df_points.columns:
        st.warning("Dataframe missing lat/lon columns.")
        return

    df = df_points[[lat_col, lon_col]].dropna().rename(columns={lat_col: "lat", lon_col: "lon"})
    if df.empty:
        st.info("No GPS points to show.")
        return

    # Try folium first for a nicer route
    try:
        import folium
        from streamlit_folium import st_folium

        # center map on mean location
        center = [df["lat"].mean(), df["lon"].mean()]
        m = folium.Map(location=center, zoom_start=zoom_start)

        # add polyline route
        coords = df[["lat", "lon"]].values.tolist()
        folium.PolyLine(locations=coords, color="blue", weight=3, opacity=0.8).add_to(m)

        # add start/end markers
        folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

        # display
        st_folium(m, width=700, height=450)

    except Exception:
        # fallback: use st.map for simple point plotting
        st.info("Folium not available — using Streamlit's simple map (install folium + streamlit-folium for better maps).")
        st.map(df)
